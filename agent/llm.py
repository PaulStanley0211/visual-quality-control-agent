"""The reasoning LLM, behind a swappable provider interface.

Only *narrative/interpretation* flows through the LLM — every disposition and the
random/systematic call are deterministic (see ``agent.decisions``). The default
``StubProvider`` is fully offline and deterministic, so the whole system runs and
all tests pass with no API key, GPU, or Ollama. Anthropic and Ollama plug in via
``settings.llm_provider`` and the optional ``llm`` dependency group.
"""
from __future__ import annotations

import json
from typing import Protocol, runtime_checkable

from config import settings
from contracts.models import ReasoningOutput


@runtime_checkable
class LLMProvider(Protocol):
    name: str

    def reason(self, facts: dict) -> ReasoningOutput:
        """Given structured inspection facts, return a probable cause + plain-language summary."""
        ...


def _pct(x: float) -> str:
    return f"{x:.0%}"


def _build_prompt(facts: dict) -> str:
    """Prompt for the real providers; the stub uses the same facts directly."""
    return (
        "You are a manufacturing quality engineer. Given the inspection facts below, write a JSON object "
        'with exactly two string keys: "probable_cause" (one sentence on the likely root cause, grounded in '
        'the machine/batch history) and "summary" (one short plain-language paragraph a reviewer can act on). '
        "Do not change the disposition or the random/systematic classification — those are already decided.\n\n"
        f"FACTS:\n{json.dumps(facts, indent=2)}\n\nRespond with only the JSON object."
    )


class StubProvider:
    """Deterministic narrative generation from the structured facts (no model call)."""

    name = "stub"

    def reason(self, facts: dict) -> ReasoningOutput:
        part_id = facts["part_id"]
        if not facts["is_defective"]:
            return ReasoningOutput(
                probable_cause="No anomaly detected; the part is within tolerance.",
                summary=(
                    f"Part {part_id} passed inspection — no defect detected "
                    f"(perception confidence {_pct(facts['detect_confidence'])}). No action required."
                ),
            )

        machine, batch = facts["machine"], facts["batch"]
        defect = facts.get("defect_type") or "surface anomaly"
        location = facts.get("location") or "an unspecified region"
        disposition = facts["disposition"]

        if facts["fault_pattern"] == "systematic":
            # Consume the already-decided drivers (single source of truth) rather than re-thresholding.
            drivers = facts.get("drivers", [])
            phrases = []
            if "machine" in drivers:
                overdue = " and maintenance is overdue" if facts.get("machine_overdue") else ""
                phrases.append(f"machine {machine['name']} is running a {_pct(machine['rate'])} recent defect rate{overdue}")
            if "batch" in drivers:
                phrases.append(
                    f"material lot {batch['material_lot']} (batch {batch['id']}) shows {_pct(batch['rate'])} recent defects"
                )
            if not phrases:
                phrases.append("the recent defect history is elevated")
            cause = (
                "Systematic process issue: "
                + "; and ".join(phrases)
                + f". The {defect} at {location} is consistent with a process fault rather than an isolated event."
            )
        else:
            cause = (
                f"Isolated (random) defect: a {defect} at {location}, while the machine "
                f"({_pct(machine['rate'])}) and batch ({_pct(batch['rate'])}) histories are within normal limits."
            )

        action_phrases = {
            "reject": "the part is rejected",
            "rework": "the part is sent for rework",
            "pass": "the part passes",
        }
        summary = (
            f"Part {part_id}: a {defect} was detected at {location} "
            f"(perception confidence {_pct(facts['detect_confidence'])}). "
            f"History indicates a {facts['fault_pattern']} fault, so {action_phrases.get(disposition, disposition)}. "
            f"{cause}"
        )
        if facts.get("escalated"):
            summary += " Confidence is below threshold, so this case is escalated to a human reviewer."
        return ReasoningOutput(probable_cause=cause, summary=summary)


class AnthropicProvider:
    """Hosted Anthropic reasoning. Requires the `llm` extra and ANTHROPIC_API_KEY."""

    name = "anthropic"

    def reason(self, facts: dict) -> ReasoningOutput:
        import anthropic  # lazy: only needed when this provider is selected

        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=settings.anthropic_model,
            max_tokens=400,
            messages=[{"role": "user", "content": _build_prompt(facts)}],
        )
        text = "".join(block.text for block in msg.content if getattr(block, "type", None) == "text")
        return _parse_reasoning(text)


class OllamaProvider:
    """Local, offline Ollama reasoning. Requires the `llm` extra and a running Ollama server."""

    name = "ollama"

    def reason(self, facts: dict) -> ReasoningOutput:
        import ollama  # lazy

        resp = ollama.chat(
            model=settings.ollama_model,
            messages=[{"role": "user", "content": _build_prompt(facts)}],
            format="json",
        )
        return _parse_reasoning(resp["message"]["content"])


def _parse_reasoning(text: str) -> ReasoningOutput:
    """Extract and validate the JSON object from a model response."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"LLM response did not contain a JSON object: {text[:200]!r}")
    return ReasoningOutput.model_validate_json(text[start : end + 1])


def get_provider(name: str | None = None) -> LLMProvider:
    name = (name or settings.llm_provider).lower()
    if name == "stub":
        return StubProvider()
    if name == "anthropic":
        return AnthropicProvider()
    if name == "ollama":
        return OllamaProvider()
    raise ValueError(f"Unknown llm_provider {name!r}; expected 'stub', 'anthropic', or 'ollama'.")
