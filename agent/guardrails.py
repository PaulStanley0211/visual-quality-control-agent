"""Guardrails: deterministic routing for the agent graph.

Confidence-threshold routing lives here; schema validation is enforced by the
Pydantic contracts at every node boundary, and disposition rules live in
``agent.decisions``.
"""
from __future__ import annotations

from agent.state import InspectionState


def route_after_decision(state: InspectionState) -> str:
    """Low-confidence cases go to a human; everything else proceeds to act."""
    return "escalate" if state.get("escalated") else "act"
