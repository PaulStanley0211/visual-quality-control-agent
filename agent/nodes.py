"""LangGraph nodes for the inspection loop: detect -> gather_context -> investigate
-> decide -> reason -> act | escalate.

Deterministic dispositions come from ``agent.decisions``; only the narrative comes
from the LLM (``deps.provider``). Every node appends to the reasoning trace, and the
act/escalate nodes persist the full audit trail to the MES.
"""
from __future__ import annotations

import logging
from datetime import date

from agent import decisions
from agent.state import AgentDeps, InspectionState
from agent.tools import flag_machine, open_capa, raise_ncr
from config import settings
from contracts.models import Actions, Decision, Diagnosis, FaultPattern, InspectionOutput
from memory import mes

logger = logging.getLogger(__name__)


def _is_overdue(last_maintenance: str | None) -> bool:
    if not last_maintenance:
        return False
    try:
        return date.fromisoformat(last_maintenance) < date.fromisoformat(settings.maintenance_cutoff)
    except ValueError:
        return False


# --- detect ---

def make_detect_node(deps: AgentDeps):
    def detect(state: InspectionState) -> dict:
        dr = state.get("detect_result")
        if dr is None:
            image_path = state.get("image_path")
            if not image_path:
                raise ValueError(f"Part '{state['part_id']}': no detect_result and no image_path provided.")
            dr = deps.get_detector().detect(image_path, part_id=state["part_id"])
        return {
            "detect_result": dr,
            "reasoning_trace": [
                f"Perception: defective={dr.is_defective}, confidence={dr.confidence:.2f}, score={dr.anomaly_score:.3f}."
            ],
        }

    return detect


# --- assess_drift (input-distribution OOD gate) ---

def make_assess_drift_node(deps: AgentDeps):
    def assess_drift(state: InspectionState) -> dict:
        if not settings.drift_enabled:
            return {}
        monitor = deps.get_drift_monitor()
        if monitor is None:
            return {}  # feature off / artifact absent -> drift stays None
        image_path = state.get("image_path")
        if not image_path:
            return {"reasoning_trace": ["Drift: not assessed (no image provided)."]}
        try:
            dr = monitor.score(image_path)
        except Exception as e:  # noqa: BLE001 - the gate's availability must never abort an inspection
            logger.warning("Drift scoring failed for part '%s': %s", state["part_id"], e)
            return {"reasoning_trace": [f"Drift: unavailable ({e})."]}
        return {
            "drift": dr,
            "reasoning_trace": [f"Drift: {dr.note} (score {dr.drift_score:.3f}, OOD={dr.is_ood})."],
        }

    return assess_drift


# --- gather_context (read long-term memory) ---

def make_gather_context_node(deps: AgentDeps):
    def gather_context(state: InspectionState) -> dict:
        conn = deps.connect()
        try:
            ctx = mes.get_part_context(state["part_id"], conn=conn)
        finally:
            conn.close()
        m, b = ctx["machine"], ctx["batch"]
        return {
            "context": ctx,
            "reasoning_trace": [
                f"MES context: machine {m['id']} ({m['name']}), batch {b['id']} ({b['material_lot']}), "
                f"operator {ctx['operator']['id']}."
            ],
        }

    return gather_context


# --- investigate (deterministic systematic-vs-random) ---

def investigate(state: InspectionState) -> dict:
    ctx = state["context"]
    m_rate = ctx["machine"]["rate"]
    b_rate = ctx["batch"]["rate"]
    thr = settings.systematic_defect_rate
    pattern = decisions.classify_fault_pattern(m_rate, b_rate, thr)
    diag_conf = decisions.diagnosis_confidence(m_rate, b_rate, thr)
    drivers = [d for d, r in (("machine", m_rate), ("batch", b_rate)) if r >= thr]
    return {
        "investigation": {
            "machine_rate": m_rate,
            "batch_rate": b_rate,
            "fault_pattern": pattern.value,
            "diagnosis_confidence": diag_conf,
            "drivers": drivers,
        },
        "reasoning_trace": [
            f"Investigation: machine rate {m_rate:.0%}, batch rate {b_rate:.0%} -> {pattern.value} "
            f"(diagnosis confidence {diag_conf:.2f})."
        ],
    }


# --- decide (deterministic disposition + escalation flag) ---

def decide(state: InspectionState) -> dict:
    dr = state["detect_result"]
    inv = state["investigation"]
    pattern = FaultPattern(inv["fault_pattern"])
    severe = decisions.is_severe(dr.defect_area, settings.severe_area_threshold)
    severity_unknown = dr.is_defective and dr.defect_area is None
    disposition = decisions.decide_disposition(dr.is_defective, pattern, severe)

    # Diagnosis ambiguity only drives escalation when the random/systematic call actually
    # changes the disposition (i.e. a non-severe defect). Otherwise the pattern is irrelevant.
    pattern_relevant = decisions.pattern_affects_disposition(dr.is_defective, severe)
    eff_diag_conf = inv["diagnosis_confidence"] if pattern_relevant else 1.0
    conf_low = decisions.should_escalate(dr.confidence, eff_diag_conf, settings.confidence_threshold)
    drift = state.get("drift")
    drift_ood = bool(drift and drift.is_ood)
    escalated = conf_low or severity_unknown or drift_ood
    routing_conf = min(dr.confidence, eff_diag_conf)

    trace = f"Decision: {disposition.value} (routing confidence {routing_conf:.2f})."
    if escalated:
        reasons = []
        if conf_low:
            reasons.append("confidence below threshold")
        if severity_unknown:
            reasons.append("unknown severity (no anomaly extent)")
        if drift_ood:
            reasons.append("image out-of-distribution (drift)")
        trace += " Escalate to human: " + "; ".join(reasons) + "."
    return {
        "decision": Decision(disposition=disposition, confidence=round(routing_conf, 4)),
        "escalated": escalated,
        "reasoning_trace": [trace],
    }


# --- reason (LLM narrative + assemble Diagnosis) ---

def make_reason_node(deps: AgentDeps):
    def reason(state: InspectionState) -> dict:
        dr = state["detect_result"]
        ctx = state["context"]
        inv = state["investigation"]
        decision = state["decision"]
        defect_label = decisions.derive_defect_label(dr.defect_area, settings.severe_area_threshold)

        facts = {
            "part_id": state["part_id"],
            "is_defective": dr.is_defective,
            "defect_type": defect_label if dr.is_defective else None,
            "location": dr.location,
            "fault_pattern": inv["fault_pattern"],
            "drivers": inv.get("drivers", []),  # single source of truth for which dimension(s) are systematic
            "disposition": decision.disposition.value,
            "escalated": state["escalated"],
            "detect_confidence": dr.confidence,
            "diagnosis_confidence": inv["diagnosis_confidence"],
            "systematic_threshold": settings.systematic_defect_rate,
            "machine_overdue": _is_overdue(ctx["machine"]["last_maintenance"]),
            "machine": {
                "name": ctx["machine"]["name"],
                "status": ctx["machine"]["status"],
                "last_maintenance": ctx["machine"]["last_maintenance"],
                "rate": ctx["machine"]["rate"],
            },
            "batch": {"id": ctx["batch"]["id"], "material_lot": ctx["batch"]["material_lot"], "rate": ctx["batch"]["rate"]},
        }
        # The decision is deterministic regardless; a provider failure must not abort the inspection.
        try:
            reasoning = deps.provider.reason(facts)
            provider_name = deps.provider.name
        except Exception as e:  # noqa: BLE001 - any provider/parse error falls back to the offline stub
            from agent.llm import StubProvider

            logger.warning("LLM provider '%s' failed (%s); falling back to stub narrative.", deps.provider.name, e)
            reasoning = StubProvider().reason(facts)
            provider_name = f"{deps.provider.name}->stub"

        diagnosis = None
        if dr.is_defective:
            diagnosis = Diagnosis(
                defect_type=defect_label,
                location=dr.location or "unspecified",
                fault_pattern=pattern_from(inv),
                probable_cause=reasoning.probable_cause,
                # Diagnosis confidence reflects BOTH perception and the random/systematic margin.
                confidence=round(min(dr.confidence, inv["diagnosis_confidence"]), 4),
            )
        return {
            "diagnosis": diagnosis,
            "summary": reasoning.summary,
            "reasoning_trace": [f"Reasoning ({provider_name}): {reasoning.probable_cause}"],
        }

    return reason


def pattern_from(inv: dict) -> FaultPattern:
    return FaultPattern(inv["fault_pattern"])


# --- act (execute corrective tools + persist) ---

def make_act_node(deps: AgentDeps):
    def act(state: InspectionState) -> dict:
        dr = state["detect_result"]
        ctx = state["context"]
        inv = state["investigation"]
        decision = state["decision"]
        plan = decisions.plan_actions(decision.disposition, pattern_from(inv), escalated=False)

        defect_label = (
            decisions.derive_defect_label(dr.defect_area, settings.severe_area_threshold) if dr.is_defective else None
        )
        new_trace: list[str] = []
        ncr_id = capa_id = None
        machine_flag = False
        conn = deps.connect()
        try:
            # One transaction: NCR + CAPA + flag + audit row are all-or-nothing (commit=False,
            # `with conn` commits on success and rolls back on any error).
            with conn:
                if plan["ncr"]:
                    ncr_id, msg = raise_ncr(conn, state["part_id"], defect_label, state.get("summary", ""), commit=False)
                    new_trace.append(msg)
                if plan["capa"]:
                    capa_id, msg = open_capa(conn, ctx["machine"]["id"], "systematic defect pattern", state.get("summary", ""), commit=False)
                    new_trace.append(msg)
                if plan["machine_flag"]:
                    new_trace.append(flag_machine(conn, ctx["machine"]["id"], commit=False))
                    machine_flag = True
                if not new_trace:
                    new_trace.append("No corrective actions required.")
                actions = Actions(ncr=plan["ncr"], capa=plan["capa"], machine_flag=machine_flag, ncr_id=ncr_id, capa_id=capa_id)
                _record(conn, state, decision, actions, escalated=False)
        finally:
            conn.close()
        output = _build_output(state, decision, actions, escalated=False, extra_trace=new_trace)
        return {"actions": actions, "output": output, "reasoning_trace": new_trace}

    return act


# --- escalate (human handoff; hold automated actions) ---

def make_escalate_node(deps: AgentDeps):
    def escalate(state: InspectionState) -> dict:
        decision = state["decision"]
        actions = Actions()  # all held
        new_trace = ["Escalated to human reviewer; automated actions held."]
        conn = deps.connect()
        try:
            with conn:
                _record(conn, state, decision, actions, escalated=True)
        finally:
            conn.close()
        output = _build_output(state, decision, actions, escalated=True, extra_trace=new_trace)
        return {"actions": actions, "output": output, "reasoning_trace": new_trace}

    return escalate


# --- helpers ---

def _build_output(
    state: InspectionState, decision: Decision, actions: Actions, escalated: bool, extra_trace: list[str]
) -> InspectionOutput:
    full_trace = list(state.get("reasoning_trace", [])) + list(extra_trace)
    dr = state.get("detect_result")
    return InspectionOutput(
        part_id=state["part_id"],
        decision=decision,
        diagnosis=state.get("diagnosis"),
        actions=actions,
        escalated=escalated,
        summary=state.get("summary", ""),
        heatmap_path=dr.heatmap_path if dr else None,
        drift=state.get("drift"),
        reasoning_trace=full_trace,
    )


def _record(conn, state: InspectionState, decision: Decision, actions: Actions, escalated: bool) -> None:
    dr = state["detect_result"]
    drift = state.get("drift")
    inv = state.get("investigation", {})
    diagnosis = state.get("diagnosis")
    mes.record_inspection(
        conn,
        part_id=state["part_id"],
        is_defective=dr.is_defective,
        confidence=dr.confidence,
        anomaly_score=dr.anomaly_score,
        drift_score=drift.drift_score if drift else None,
        defect_type=diagnosis.defect_type if diagnosis else None,
        disposition=decision.disposition.value,
        fault_pattern=inv.get("fault_pattern"),
        escalated=escalated,
        reasoning=state.get("summary", ""),
        actions=actions.model_dump(),
        source="agent",  # excluded from recent_defect_rate; never feeds back into classification
        commit=False,  # caller's `with conn` transaction commits once
    )
