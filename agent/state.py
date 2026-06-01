"""Working memory: the per-inspection state carried through the LangGraph loop."""
from __future__ import annotations

import operator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Optional, TypedDict

from agent.llm import LLMProvider, get_provider
from contracts.models import Actions, Decision, DetectResult, Diagnosis, DriftResult, InspectionOutput


class InspectionState(TypedDict, total=False):
    """State flowing through the graph. ``reasoning_trace`` accumulates across nodes."""

    # --- inputs ---
    part_id: str
    image_path: Optional[str]
    detect_result: Optional[DetectResult]  # may be pre-injected (scenarios / tests) to skip perception
    drift: Optional[DriftResult]  # input-distribution drift assessment (None if not assessed)

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
    _drift_monitor: Any = None
    _drift_tried: bool = False

    def get_detector(self):
        if self._detector is None:
            from perception.detector import Detector

            self._detector = Detector()
        return self._detector

    def get_drift_monitor(self):
        """Lazily build the DriftMonitor; return None (feature off) if its artifacts are absent
        or it fails to load. Memoized so a missing artifact isn't retried every inspection."""
        if self._drift_monitor is not None:
            return self._drift_monitor
        if self._drift_tried:
            return None
        self._drift_tried = True
        try:
            from drift.monitor import DriftMonitor

            self._drift_monitor = DriftMonitor()
        except Exception as e:  # noqa: BLE001 - missing/invalid artifact => drift simply off
            import logging

            logging.getLogger(__name__).info("Drift monitor unavailable; drift disabled (%s).", e)
            self._drift_monitor = None
        return self._drift_monitor

    def connect(self):
        from memory import mes

        return mes.connect(self.db_path)
