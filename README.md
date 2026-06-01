# Visual Quality Control Agent

Autonomous single-station visual quality control. It detects part defects, diagnoses whether a fault is **random or systematic** from factory history, decides a disposition (**pass / rework / reject**), triggers corrective workflows, and escalates low-confidence cases to a human — a goal-driven **plan → act → observe → act** loop, with both perception and reasoning held to defined accuracy budgets.

> Built on the [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) dataset. Runs fully offline — no API key or GPU required; a hosted LLM is optional. All milestones complete and validated on **two** product categories.

## How it works

```
image + part_id
      │
      ▼
  PatchCore detect ──►  is_defective · confidence · heatmap
      │
      ▼
  LangGraph agent:  detect → gather context → investigate → decide → reason → act | escalate
      │                       (reads machine / batch / operator history from a SQLite MES)
      ▼
  Decision · Diagnosis · Actions   (+ plain-language summary, full audit trail)
```

- **Perception** — PatchCore anomaly detector ([anomalib](https://github.com/open-edge-platform/anomalib)) behind one `detect(image) → DetectResult` interface.
- **Agent** — a [LangGraph](https://github.com/langchain-ai/langgraph) state machine. Every disposition and the random-vs-systematic call are **deterministic pure functions** ([agent/decisions.py](agent/decisions.py)); the LLM only writes the narrative.
- **Memory** — per-inspection graph state plus a long-term **SQLite MES** of part / machine / batch / operator history that drives the systematic-vs-random reasoning.
- **Guardrails** — confidence-threshold escalation, Pydantic-validated outputs, and a full audit trail. The agent's own decisions are excluded from the defect-rate signal, so it can't amplify itself into a false "systematic" verdict.

## Results

**Perception** (PatchCore on MVTec AD, CPU) — the same pipeline, switched by config:

| Category | Image AUROC | Published baseline | Holdout FAR | Holdout FRR |
|---|---|---|---|---|
| `bottle` | **0.9992** | ~1.000 | 3.2% (1/31) | 0.0% |
| `hazelnut` | **0.9982** | ~1.000 | 2.9% (1/35) | 5.0% |

Thresholds are calibrated on a seeded split and FAR/FRR reported on a disjoint **holdout** (an honest generalization estimate, not in-sample). Baselines are PatchCore's (Roth et al., 2022). With few defect samples a 2% false-accept rate can't be *certified* on held-out data — it's reported with FAR granularity and a Wilson interval (see [Limitations](#limitations)).

**Agent** (offline stub, 12 labeled scenarios incl. boundary, both-driver, and escalation cases):

| Metric | Result | Target |
|---|---|---|
| Disposition accuracy | **100%** | ≥ 95% |
| Random-vs-systematic accuracy | **100%** | ≥ 95% |
| Escalation & corrective-action accuracy | **100%** | — |

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) (it provisions Python 3.11 automatically).

```bash
uv sync                                      # install locked deps (CPU torch)

# Perception — fetch data, fit PatchCore, calibrate the error budget
uv run python -m perception.train
uv run python -m eval.perception_eval

# Agent — seed the MES, score the scenario set
uv run python -m memory.seed
uv run python -m eval.agent_eval

uv run python -m pytest                       # 88 tests

# Serve / demo
uv run uvicorn service.app:app --port 8000        # POST /inspect, GET /health, GET /drift
uv run streamlit run ui/streamlit_app.py          # interactive demo
docker build -t vqc-agent -f service/Dockerfile . && docker run -p 8000:8000 vqc-agent
```

> **Data:** anomalib's MVTec downloader is offline, so [perception/prepare_data.py](perception/prepare_data.py) fetches just the active category from a public Hugging Face mirror. MVTec AD is **CC BY-NC-SA 4.0 (non-commercial)** — downloaded locally, never redistributed (`datasets/` is git-ignored).

## Configuration

Every value is env-overridable via `VQC_*` (or a local `.env`):

| Variable | Default | Purpose |
|---|---|---|
| `VQC_CATEGORY` | `bottle` | Active product; per-category models coexist under `artifacts/perception/<category>/` |
| `VQC_LLM_PROVIDER` | `stub` | `stub` (offline) · `anthropic` · `ollama` |
| `ANTHROPIC_API_KEY` | — | Enables the hosted Claude narrative when the provider is `anthropic` |
| `VQC_CONFIDENCE_THRESHOLD` | `0.60` | Below this, a case escalates to a human |
| `VQC_CORESET_SAMPLING_RATIO` | `0.1` | PatchCore memory-bank fraction (1% gives ≈ the same AUROC ~10× faster) |

**Use a hosted LLM** — the default stub makes the whole system run with no key/GPU. To use Claude for richer narration (the disposition stays deterministic regardless), add a `.env`:

```
ANTHROPIC_API_KEY=sk-ant-...
VQC_LLM_PROVIDER=anthropic
```

**Add another category** — set `VQC_CATEGORY=<name>` and re-run `perception.train` + `eval.perception_eval`. Data fetch, layout, and artifacts generalize automatically; the trained bottle model is left untouched.

## Project structure

```
perception/   PatchCore training, data prep, detect() interface
agent/        LangGraph nodes, tools, state, guardrails, decisions, llm
contracts/    Pydantic I/O contracts
memory/       SQLite MES schema + seed + query/write
eval/         perception & agent validation
service/      FastAPI app + Docker
ui/           Streamlit demo
tests/        unit, regression, escalation, reasoning
```

## Drift monitoring

A `drift/` package watches whether incoming images still resemble the training-good distribution, guarding the pass path against silent false-accepts when the camera, lighting, or part variant shifts.

- **Per-image OOD gate** — each part image is scored by kNN distance to a per-category reference embedding set (ImageNet resnet18 backbone, decoupled from anomalib). Non-defective parts that score above the calibrated threshold are escalated to a human; the drift read is annotated on every inspection regardless of escalation. Defective parts are excluded from drift escalation — a defect is naturally far from the training-good manifold, so its drift score is not a distribution-shift signal.
- **MES-backed population monitor** — per-inspection drift scores are stored in SQLite; `drift/report.py` computes a windowed PSI over the non-defective (good) stream and reports % OOD parts and image-stat trends.
- **Optional / per-category** — absent the reference artifact, the feature is simply off. Artifacts live under `artifacts/drift/<category>/`, parallel to `artifacts/perception/<category>/`, and are switched by `VQC_CATEGORY`.

**Validated numbers** (leave-one-out calibration on training-good + disjoint holdout, Wilson CI — same methodology as perception):

| Category | Separability AUROC | Calibration false-alarm | All perturbations detected |
|---|---|---|---|
| `bottle` | **1.000** | 5.3% (LOO, n=209) | **100%** (brightness↑↓, contrast, blur, noise, JPEG) |
| `hazelnut` | **0.997** | 5.1% (LOO, n=391) | 100% except blur (82.5%) |

```bash
uv run python -m drift.reference     # build per-category reference (after perception.train)
uv run python -m eval.drift_eval     # calibrate threshold → drift_metrics.json + plot
uv run python -m drift.report        # windowed PSI + %OOD report over the live MES good stream
# per category: VQC_CATEGORY=hazelnut uv run python -m drift.reference   (etc.)
```

Service: `GET /drift` returns the windowed report as JSON; `GET /health` now also reports `drift_enabled` and `drift_reference_present`.

## Limitations

- The MES is synthetic-but-realistic; the query interface is built to mirror a real MES (the production path).
- Few defect samples → a 2% false-accept rate can't be *statistically certified* on held-out data; metrics carry their sampling uncertainty.
- Single-station scope; multi-station orchestration and logical anomalies (MVTec LOCO AD) are natural extensions.
- Vision accuracy degrades under lighting / camera / part drift; the drift monitor detects image-level OOD shifts and escalates for human review, but continuous retraining is still required in production to restore validated accuracy.

## Attribution

MVTec AD dataset © MVTec Software GmbH, CC BY-NC-SA 4.0 (non-commercial). Credited, not redistributed.
