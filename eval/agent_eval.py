"""Agent validation on the labeled scenario set.

Scores disposition accuracy and random-vs-systematic accuracy (plus escalation
accuracy) against the target. Each scenario runs against a freshly-seeded MES so
the agent's own writes never leak between scenarios. Uses the offline stub LLM.

Run:  uv run python -m eval.agent_eval
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from agent.graph import build_graph, run_inspection
from agent.llm import StubProvider
from agent.state import AgentDeps
from config import settings
from contracts.models import DetectResult
from memory import seed as seed_module

SCENARIOS_PATH = Path(__file__).resolve().parent / "scenarios.yaml"


def _detect_result(spec: dict) -> DetectResult:
    defective = bool(spec["is_defective"])
    return DetectResult(
        is_defective=defective,
        confidence=float(spec["confidence"]),
        anomaly_score=0.9 if defective else 0.4,
        threshold=0.5,
        defect_area=float(spec.get("defect_area", 0.0)),
        location=spec.get("location"),
    )


def evaluate() -> dict:
    scenarios = yaml.safe_load(SCENARIOS_PATH.read_text())
    eval_db = settings.artifacts_dir / "agent_eval_mes.db"
    eval_db.parent.mkdir(parents=True, exist_ok=True)

    deps = AgentDeps(db_path=eval_db, provider=StubProvider())
    app = build_graph(deps)

    rows = []
    disp_ok = fp_ok = esc_ok = act_ok = 0
    fp_total = 0
    for s in scenarios:
        seed_module.seed(eval_db, verbose=False)  # fresh history per scenario for independence
        out = run_inspection(app, s["part_id"], detect_result=_detect_result(s["detect"]))
        exp = s["expected"]

        got_disp = out.decision.disposition.value
        got_fp = out.diagnosis.fault_pattern.value if out.diagnosis else None
        got_esc = out.escalated
        got_act = {"ncr": out.actions.ncr, "capa": out.actions.capa, "machine_flag": out.actions.machine_flag}

        disp_match = got_disp == exp["disposition"]
        esc_match = got_esc == exp["escalated"]
        act_match = got_act == exp["actions"]
        disp_ok += disp_match
        esc_ok += esc_match
        act_ok += act_match
        if exp.get("fault_pattern") is not None:
            fp_total += 1
            fp_ok += got_fp == exp["fault_pattern"]

        rows.append({
            "name": s["name"], "part_id": s["part_id"],
            "disposition": f"{got_disp}{'' if disp_match else ' != ' + exp['disposition']}",
            "fault_pattern": f"{got_fp}{'' if got_fp == exp.get('fault_pattern') else ' != ' + str(exp.get('fault_pattern'))}",
            "escalated": f"{got_esc}{'' if esc_match else ' != ' + str(exp['escalated'])}",
            "actions": "ok" if act_match else f"{got_act} != {exp['actions']}",
            "ok": bool(disp_match and esc_match and act_match and (exp.get("fault_pattern") is None or got_fp == exp["fault_pattern"])),
        })

    n = len(scenarios)
    metrics = {
        "n_scenarios": n,
        "disposition_accuracy": round(disp_ok / n, 4),
        "fault_pattern_accuracy": round(fp_ok / fp_total, 4) if fp_total else None,
        "escalation_accuracy": round(esc_ok / n, 4),
        "actions_accuracy": round(act_ok / n, 4),
        "target": settings.agent_accuracy_target,
        "disposition_target_met": disp_ok / n >= settings.agent_accuracy_target,
        "fault_pattern_target_met": (fp_ok / fp_total >= settings.agent_accuracy_target) if fp_total else None,
        "actions_target_met": act_ok / n >= settings.agent_accuracy_target,
    }

    out_path = settings.artifacts_dir / "agent_metrics.json"
    out_path.write_text(json.dumps({"metrics": metrics, "scenarios": rows}, indent=2))

    print(f"[agent_eval] {n} scenarios (provider=stub):")
    for r in rows:
        mark = "OK " if r["ok"] else "XX "
        print(f"  {mark}{r['name']:36} disp={r['disposition']:18} pattern={r['fault_pattern']:16} "
              f"esc={r['escalated']:6} actions={r['actions']}")
    print(f"[agent_eval] disposition {metrics['disposition_accuracy']:.0%}, "
          f"random/systematic {metrics['fault_pattern_accuracy']:.0%}, "
          f"escalation {metrics['escalation_accuracy']:.0%}, actions {metrics['actions_accuracy']:.0%} "
          f"(target {settings.agent_accuracy_target:.0%}).")
    return metrics


if __name__ == "__main__":
    evaluate()
