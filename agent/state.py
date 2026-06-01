"""Working memory: the per-inspection state carried through the LangGraph loop."""
from __future__ import annotations

import operator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Optional, TypedDict

from agent.llm import LLMProvider, get_provider
from contracts.models import Actions, Decision, DetectResult, Diagnosis, InspectionOutput


class InspectionState(TypedDict, total=False):
    """State flowing through the graph. ``reasoning_trace`` accumulates across nodes."""

    # --- inputs ---
    part_id: str
    image_path: Optional[str]
    detect_result: Optional[DetectResult]  # may be pre-injected (scenarios / tests) to skip perception

    # --- gathered / derived ---
    context: dict
    investigation: dict  # machine_rate, batch_rate, fault_pattern, diagnosis_confidence, drivers

    # --- decisions / outputs ---
    decision: Optional[Decision]
    diagnosis: Optional[Diagnosis]
    actions: Actions
    escalated: bool
    summary: str
    output: Optional[InspectionOutput]

    reasoning_trace: Annotated[list[str], operator.add]


@dataclass
class AgentDeps:
    """Injected dependencies so the graph is testable and the heavy detector is lazy."""

    db_path: Optional[Path] = None
    provider: LLMProvider = field(default_factory=get_provider)
    _detector: Any = None

    def get_detector(self):
        if self._detector is None:
            from perception.detector import Detector

            self._detector = Detector()
        return self._detector

    def connect(self):
        from memory import mes

        return mes.connect(self.db_path)
