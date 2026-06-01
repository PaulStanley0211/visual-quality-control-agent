"""Schema-validated I/O contracts shared across perception, agent, and service.

Milestone A defines :class:`DetectResult` (perception output). Milestone B
extends this module with the agent-side diagnosis / decision / action contracts.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class DetectResult(BaseModel):
    """Output of the perception layer for a single part image."""

    is_defective: bool = Field(description="True if anomaly_score >= threshold.")
    confidence: float = Field(ge=0.0, le=1.0, description="Calibrated confidence in the is_defective call.")
    anomaly_score: float = Field(description="Raw image-level anomaly score from PatchCore.")
    threshold: float = Field(description="Operating threshold (calibrated for FAR <= target).")
    heatmap_path: str | None = Field(default=None, description="Path to the saved anomaly heatmap overlay, if written.")
