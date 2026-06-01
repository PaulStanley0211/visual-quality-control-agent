"""Schema-validated I/O contracts shared across perception, agent, and service.

Milestone A defines :class:`DetectResult` (perception output). Milestone B adds
the agent-side diagnosis / decision / action contracts and the three-part
:class:`InspectionOutput` that is the agent's external result.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class DetectResult(BaseModel):
    """Output of the perception layer for a single part image."""

    is_defective: bool = Field(description="True if anomaly_score >= threshold.")
    confidence: float = Field(ge=0.0, le=1.0, description="Calibrated confidence in the is_defective call.")
    anomaly_score: float = Field(description="Raw image-level anomaly score from PatchCore.")
    threshold: float = Field(description="Operating threshold (calibrated for FAR <= target).")
    heatmap_path: str | None = Field(default=None, description="Path to the saved anomaly heatmap overlay, if written.")
    defect_area: float | None = Field(default=None, description="Fraction of the heatmap above the hot threshold (anomaly extent).")
    location: str | None = Field(default=None, description="Region of the part where the anomaly concentrates (from the heatmap).")


class Disposition(str, Enum):
    """Final routing decision for a part."""

    PASS = "pass"
    REWORK = "rework"
    REJECT = "reject"


class FaultPattern(str, Enum):
    """Whether a defect looks like an isolated event or a process problem."""

    RANDOM = "random"
    SYSTEMATIC = "systematic"


class Diagnosis(BaseModel):
    """Interpretation of a detected defect (only present when a defect is found)."""

    defect_type: str = Field(description="Coarse defect descriptor (from perception / annotation).")
    location: str = Field(description="Where on the part the anomaly is concentrated.")
    fault_pattern: FaultPattern = Field(description="random vs systematic — decided deterministically from history.")
    probable_cause: str = Field(description="Plain-language likely cause (LLM-interpreted from MES context).")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the diagnosis.")


class Decision(BaseModel):
    """Deterministic disposition plus the confidence that drove routing."""

    disposition: Disposition
    confidence: float = Field(ge=0.0, le=1.0)


class Actions(BaseModel):
    """Corrective actions the agent took (or would take), with any created record ids."""

    ncr: bool = Field(default=False, description="Non-conformance report raised.")
    capa: bool = Field(default=False, description="Corrective/preventive-action ticket created.")
    machine_flag: bool = Field(default=False, description="Machine flagged for attention.")
    ncr_id: str | None = None
    capa_id: str | None = None


class ReasoningOutput(BaseModel):
    """The narrow slice of judgment delegated to the LLM (everything else is deterministic)."""

    probable_cause: str = Field(description="Likely cause given the defect and the machine/batch/operator context.")
    summary: str = Field(description="One short plain-language paragraph a human reviewer can act on.")


class InspectionOutput(BaseModel):
    """The agent's three-part result: Decision, Diagnosis, Actions, plus a plain-language summary."""

    part_id: str
    decision: Decision
    diagnosis: Diagnosis | None = Field(default=None, description="None when the part passes with no defect.")
    actions: Actions = Field(default_factory=Actions)
    escalated: bool = Field(default=False, description="True if routed to a human reviewer for low confidence.")
    summary: str
    heatmap_path: str | None = Field(default=None, description="Anomaly heatmap overlay produced by this inspection.")
    reasoning_trace: list[str] = Field(default_factory=list, description="Ordered audit trail of the agent's steps.")
