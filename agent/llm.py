"""The reasoning LLM, behind a swappable provider interface.

Only *narrative/interpretation* flows through the LLM — every disposition and the
random/systematic call are deterministic (see ``agent.decisions``). The default
``StubProvider`` is fully offline and deterministic, so the whole system runs and
all tests pass with no API key, GPU, or Ollama. Anthropic and Ollama plug in via
``settings.llm_provider`` and the optional ``llm`` dependency group.

Provider failures never abort an inspection: ``agent.nodes.make_reason_node`` wraps
``provider.reason`` and falls back to the stub narrative on any exception.
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


# --- shared prompt (stable across every request → cacheable prefix for hosted providers) ---

# Kept byte-stable on purpose: this is the cache prefix. Do NOT interpolate per-request
# values (part ids, dates, rates) into it — those belong in the user turn (`_facts_block`),
# after the cache breakpoint. See shared/prompt-caching.md.
SYSTEM_PROMPT = """You are a senior manufacturing quality engineer writing the narrative portion of an \
automated visual-inspection report for a single part.

You are given the inspection FACTS as a JSON object. Two of those facts — the `disposition` \
(pass/rework/reject) and the `fault_pattern` (random vs systematic) — were already decided upstream \
by deterministic logic and an error-budgeted detector. They are final. Do NOT change, second-guess, \
hedge, or recompute them; explain them.

Write exactly two fields:
- probable_cause: one or two sentences naming the most likely root cause. Ground it ONLY in the \
machine/batch history present in the facts. If `fault_pattern` is "systematic", attribute it to the \
driver(s) listed in `drivers` (machine and/or batch) and cite the relevant recent defect rate; mention \
overdue maintenance only when `machine_overdue` is true. If "random", say the histories are within \
normal limits and the defect looks isolated.
- summary: one short plain-language paragraph (2-4 sentences) a line supervisor can act on. State the \
part id, the defect and where it is, the disposition and why, and — if `escalated` is true — that the \
case is being escalated to a human reviewer with automated actions held.

Rules:
- Never invent defect types, locations, rates, machines, batches, or causes that are not in the facts.
- Use plain industrial language, not marketing tone. Be specific and concise; no preamble.
- Report rates as percentages. Do not output anything except the two fields."""


def _facts_block(facts: dict) -> str:
    """The per-request user turn: the facts to interpret (varies every call; sits after the cache breakpoint)."""
    return f"Inspection FACTS:\n{json.dumps(facts, indent=2)}"


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
    """Hosted Anthropic reasoning. Requires the `llm` extra and ANTHROPIC_API_KEY.

    Uses structured outputs (``messages.parse`` against ``ReasoningOutput``) so the response is
    schema-valid by construction — no brittle JSON scraping. The stable instructions live in a cached
    ``system`` block; only the per-part facts vary between calls.
    """

    name = "anthropic"

    def __init__(self) -> None:
        self._client = None  # lazily constructed so stub-only runs never import anthropic

    def _get_client(self):
        if self._client is None:
            import anthropic  # lazy: only needed when this provider is actually selected

            # api_key=None lets the SDK self-resolve from the environment; settings reads it from
            # ANTHROPIC_API_KEY (env or .env), so a key placed in .env works without exporting it.
            self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        return self._client

    def reason(self, facts: dict) -> ReasoningOutput:
        client = self._get_client()
        response = client.messages.parse(
            model=settings.anthropic_model,
            max_tokens=600,  # two short fields; narrative is a sentence + a short paragraph
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    # Caches the instruction prefix. NB: Opus 4.8's minimum cacheable prefix is ~4096
                    # tokens, so this prompt is below threshold and won't actually cache today — the
                    # marker is correct/forward-safe and engages automatically if the prompt grows.
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": _facts_block(facts)}],
            output_format=ReasoningOutput,  # forces schema-valid {probable_cause, summary}
        )
        out = response.parsed_output
        if out is None:  # refusal or truncation — let the node fall back to the stub
            raise ValueError(f"Anthropic returned no parsed output (stop_reason={response.stop_reason}).")
        return out


class OllamaProvider:
    """Local, offline Ollama reasoning. Requires the `llm` extra and a running Ollama server."""

    name = "ollama"

    def reason(self, facts: dict) -> ReasoningOutput:
        import ollama  # lazy

        resp = ollama.chat(
            model=settings.ollama_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _facts_block(facts) + '\n\nRespond with only a JSON object: '
                 '{"probable_cause": "...", "summary": "..."}.'},
            ],
            format="json",
        )
        return _parse_reasoning(resp["message"]["content"])


def _parse_reasoning(text: str) -> ReasoningOutput:
    """Extract and validate the JSON object from a model response (used by the Ollama path)."""
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
