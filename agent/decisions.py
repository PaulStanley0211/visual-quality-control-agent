"""Deterministic decision logic — the disposition judgment kept out of the LLM.

Every pass/rework/reject and random/systematic call lives here as a pure function,
so behaviour is fully testable and auditable. The LLM only narrates/interprets.
"""
from __future__ import annotations

from contracts.models import Disposition, FaultPattern


def classify_fault_pattern(machine_rate: float, batch_rate: float, threshold: float) -> FaultPattern:
    """Systematic if the machine OR batch recent defect rate is at/above the threshold."""
    if machine_rate >= threshold or batch_rate >= threshold:
        return FaultPattern.SYSTEMATIC
    return FaultPattern.RANDOM


def is_severe(defect_area: float | None, area_threshold: float) -> bool:
    """Severe if the anomalous area (fraction of the heatmap) is at/above the threshold.

    PatchCore is class-agnostic, so severity is derived from anomaly *extent* rather than
    a (non-existent) defect classifier.
    """
    return defect_area is not None and defect_area >= area_threshold


def derive_defect_label(defect_area: float | None, area_threshold: float) -> str:
    """A coarse, honest defect descriptor derived from anomaly extent (not a class label)."""
    if defect_area is None:
        return "surface anomaly"
    return "large-area surface defect" if defect_area >= area_threshold else "small-area surface defect"


def decide_disposition(is_defective: bool, fault_pattern: FaultPattern, severe: bool) -> Disposition:
    """Pass clean parts; reject systematic or severe defects; rework isolated minor defects."""
    if not is_defective:
        return Disposition.PASS
    if fault_pattern is FaultPattern.SYSTEMATIC:
        return Disposition.REJECT
    if severe:
        return Disposition.REJECT
    return Disposition.REWORK


def pattern_affects_disposition(is_defective: bool, severe: bool) -> bool:
    """True when the random/systematic call changes the disposition.

    A severe defect rejects either way, and a clean part passes either way — so only a
    non-severe defect's disposition depends on the fault pattern. Used so that ambiguity in
    the pattern only forces escalation when it would actually change the outcome.
    """
    return is_defective and not severe


def diagnosis_confidence(machine_rate: float, batch_rate: float, threshold: float) -> float:
    """Confidence in the random/systematic call: minimal (0.5) at the threshold, rising
    as the deciding defect rate moves away from it. A rate near the threshold is genuinely
    ambiguous and should drive escalation."""
    deciding = max(machine_rate, batch_rate)
    margin = abs(deciding - threshold)
    return min(0.99, 0.5 + 2.0 * margin)


def should_escalate(detect_confidence: float, diagnosis_confidence: float, threshold: float) -> bool:
    """Escalate to a human when either confidence falls below the routing threshold."""
    return detect_confidence < threshold or diagnosis_confidence < threshold


def plan_actions(disposition: Disposition, fault_pattern: FaultPattern, escalated: bool) -> dict:
    """Which corrective actions to take. Escalation holds all automated actions for the human."""
    actions = {"ncr": False, "capa": False, "machine_flag": False}
    if escalated:
        return actions
    if disposition in (Disposition.REJECT, Disposition.REWORK):
        actions["ncr"] = True
    if disposition is Disposition.REJECT and fault_pattern is FaultPattern.SYSTEMATIC:
        actions["capa"] = True
        actions["machine_flag"] = True
    return actions
