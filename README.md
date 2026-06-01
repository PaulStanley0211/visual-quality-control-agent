# Visual Quality Control Agent

An autonomous industrial inspection agent that detects part defects, diagnoses whether a fault is **random or systematic** using long-term factory history, decides a disposition (**pass / rework / reject**), and triggers corrective workflows autonomously — escalating low-confidence cases to a human. It goes beyond detect-and-report into a goal-driven **plan → act → observe → act** loop, with both perception and reasoning held to defined accuracy budgets.

> **Status:** Milestones A (Perception), B (long-term memory + LangGraph agent loop), and C (FastAPI service + Docker + Streamlit demo) complete and verified. See [Roadmap](#roadmap).

## Architecture

- **Perception** — PatchCore anomaly detector (via [anomalib](https://github.com/open-edge-platform/anomalib)) trained on the [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) dataset; a single `detect(image) → DetectResult` interface.
- **Agent** *(Milestone B)* — a LangGraph state machine: `detect → gather context → investigate → reason → decide → act / escalate`, with deterministic dispositions in code and judgment confined to a swappable LLM.
- **Memory** *(Milestone B)* — working memory (per-inspection graph state) plus a long-term SQLite MES store of part / machine / batch / operator history that powers systematic-vs-random reasoning.
- **Guardrails** — confidence-threshold routing, schema-validated (Pydantic) outputs, and a full audit trail of every decision and its reasoning.

## Stack

`uv` (exclusive, locked deps) · Python 3.11 · anomalib 2.5 / PatchCore on PyTorch (CPU) · LangGraph · Pydantic · FastAPI · Streamlit. The reasoning LLM is **offline-first and swappable**: a deterministic stub is the default so the whole system runs and all tests pass with no API key or GPU; Anthropic or a local Ollama model plug in via config.

## Setup

Requires [uv](https://docs.astral.sh/uv/). uv provisions a project-local Python 3.11 automatically.

```bash
uv sync                 # create the venv and install locked dependencies (CPU torch wheels)
```

## Perception: train, evaluate, test

```bash
uv run python -m perception.train          # fetch the 'bottle' category, fit PatchCore (CPU), export the model
uv run python -m eval.perception_eval      # AUROC + calibration/holdout FAR-FRR against the error budget
uv run python -m pytest tests/             # regression suite
```

> **Dataset note:** anomalib's built-in MVTec downloader currently 404s (its mirror is offline). `perception/prepare_data.py` fetches only the configured category (~320 MB for `bottle`) from a public Hugging Face mirror and lays it out in anomalib's expected structure. MVTec AD is **CC BY-NC-SA 4.0 (non-commercial)** — it is downloaded locally for research use and **never redistributed** (the `datasets/` folder is git-ignored).

## Perception results (MVTec `bottle`, CPU)

| Metric | Value | Notes |
|---|---|---|
| **Image-level AUROC** | **0.9992** (holdout 0.9968) | Threshold-free separability; matches the published PatchCore baseline (~1.0) |
| Held-out false-accept rate | **3.2%** (1 hard defect of 31) | = 1/31, the dataset's FAR resolution; Wilson 95% upper bound 16% |
| Held-out false-reject rate | 0.0% | No good parts rejected |
| In-sample FAR (reference) | 1.6% | Threshold fit *and* scored on the full set — optimistic |

The operating threshold is calibrated on a seeded **calibration split** and FAR/FRR are reported on a disjoint **holdout split**, so the budget claim is an honest generalization estimate rather than in-sample. The model separates classes near-perfectly; `bottle` simply has too few defect samples to *certify* a 2% false-accept rate on held-out data (a single missed defect already exceeds it). This is a dataset-size limitation, not a model weakness — see [Limitations](#limitations).

## Agent: memory, reasoning loop, evaluation

```bash
uv run python -m memory.seed         # build the synthetic MES (machines/batches/operators/parts/history)
uv run python -m eval.agent_eval     # score the labeled scenario set
```

The agent runs a **LangGraph** loop — `detect → gather context → investigate → decide → reason → act | escalate` — over a seeded **SQLite MES**. Every disposition (pass/rework/reject) and the random-vs-systematic call are **deterministic pure functions** ([agent/decisions.py](agent/decisions.py)); the LLM (default: offline stub) only writes the narrative. A defect is **systematic** when the machine *or* batch recent defect rate clears a threshold (e.g. an overdue machine or a bad material lot), otherwise **random**. Low-confidence cases — weak perception, ambiguous near-threshold history, or unknown severity — **escalate to a human** and hold automated actions. Every inspection, NCR, CAPA, and machine flag is written back to the MES as an audit trail.

A key design point surfaced by review: the agent's own decision rows are tagged `source='agent'` and **excluded** from the defect-rate signal (only `source='qc'` production history counts), so the agent can never amplify its own dispositions into a false "systematic" verdict.

**Agent validation** (offline stub, 12 labeled scenarios incl. boundary + both-driver + escalation cases):

| Metric | Result | Target |
|---|---|---|
| Disposition accuracy | **100%** | ≥ 95% |
| Random-vs-systematic accuracy | **100%** | ≥ 95% |
| Escalation accuracy | **100%** | — |
| Corrective-action accuracy | **100%** | — |

## Service & demo

```bash
# FastAPI service — POST /inspect (image + part_id) -> InspectionOutput JSON, GET /health
uv run uvicorn service.app:app --host 0.0.0.0 --port 8000

# Streamlit demo — upload an image, watch the investigation, view the three-part output
uv run streamlit run ui/streamlit_app.py

# Container (needs a trained model under artifacts/perception/ first)
docker build -t vqc-agent -f service/Dockerfile .
docker run -p 8000:8000 vqc-agent
```

The service builds the compiled graph once at startup and seeds the MES if empty; the confidence threshold and LLM provider are env-overridable (`VQC_*`) for tuning at deploy time.

## Project structure

```
perception/   PatchCore training, data prep, detect() interface
agent/        LangGraph nodes, tools, state, guardrails, decisions, llm
contracts/    Pydantic I/O contracts
memory/       SQLite MES schema + seed + query/write interface
eval/         perception & agent validation
service/      FastAPI app + Docker                              (Milestone C)
ui/           Streamlit demo                                    (Milestone C)
tests/        unit, regression, escalation, reasoning tests
```

## Roadmap

- **Milestone A — Perception** ✅ train / detect / validate against the error budget.
- **Milestone B — Memory + Agent loop** ✅ seeded SQLite MES, LangGraph agent, structured decision/diagnosis/actions with an escalation path, labeled-scenario validation (disposition + random-vs-systematic accuracy).
- **Milestone C — Service + UI** ✅ FastAPI `/inspect` endpoint, uv-locked Docker image (CPU), Streamlit demo.

## Limitations

- The MES is synthetic-but-realistic; the query interface is designed for real MES integration, which is the path to production.
- Single-station, single-category (`bottle`) scope by design; multi-station orchestration and logical-anomaly handling (MVTec LOCO AD) are natural extensions.
- The chosen dataset has few defect samples, so a 2% false-accept rate cannot be *statistically certified* on held-out data; metrics are reported with their sampling uncertainty (FAR granularity + Wilson interval).
- Vision accuracy degrades under lighting / camera / part-variant drift; continuous input monitoring and periodic retraining are required in a live deployment.

## Attribution

MVTec AD dataset © MVTec Software GmbH, licensed under CC BY-NC-SA 4.0 (non-commercial). This project credits the dataset and does not redistribute it.
