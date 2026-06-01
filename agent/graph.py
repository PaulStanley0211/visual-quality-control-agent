"""The LangGraph inspection loop: detect -> assess_drift -> gather_context -> investigate
-> decide -> reason -> (act | escalate).

``build_graph`` compiles the state machine with injected dependencies;
``run_inspection`` invokes it for one part and returns the schema-validated
:class:`InspectionOutput`.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agent import nodes
from agent.guardrails import route_after_decision
from agent.state import AgentDeps, InspectionState
from contracts.models import DetectResult, InspectionOutput


def build_graph(deps: AgentDeps | None = None):
    deps = deps or AgentDeps()
    g = StateGraph(InspectionState)
    g.add_node("detect", nodes.make_detect_node(deps))
    g.add_node("assess_drift", nodes.make_assess_drift_node(deps))
    g.add_node("gather_context", nodes.make_gather_context_node(deps))
    g.add_node("investigate", nodes.investigate)
    g.add_node("decide", nodes.decide)
    g.add_node("reason", nodes.make_reason_node(deps))
    g.add_node("act", nodes.make_act_node(deps))
    g.add_node("escalate", nodes.make_escalate_node(deps))

    g.add_edge(START, "detect")
    g.add_edge("detect", "assess_drift")
    g.add_edge("assess_drift", "gather_context")
    g.add_edge("gather_context", "investigate")
    g.add_edge("investigate", "decide")
    g.add_edge("decide", "reason")
    g.add_conditional_edges("reason", route_after_decision, {"act": "act", "escalate": "escalate"})
    g.add_edge("act", END)
    g.add_edge("escalate", END)
    return g.compile()


def run_inspection(
    app,
    part_id: str,
    image_path: str | None = None,
    detect_result: DetectResult | None = None,
) -> InspectionOutput:
    """Run one inspection through the compiled graph and return its InspectionOutput."""
    initial: InspectionState = {"part_id": part_id, "reasoning_trace": []}
    if image_path is not None:
        initial["image_path"] = image_path
    if detect_result is not None:
        initial["detect_result"] = detect_result
    final = app.invoke(initial)
    output = final.get("output")
    if output is None:
        raise RuntimeError(f"Inspection for part '{part_id}' produced no output.")
    return output
