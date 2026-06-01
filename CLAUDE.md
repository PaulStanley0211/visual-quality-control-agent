# Visual Quality Control Agent

Autonomous industrial inspection agent that detects part defects, diagnoses cause, and acts — going beyond detect-and-report into a goal-driven plan / act / observe / act loop, with both perception and reasoning held to defined accuracy budgets.

## Use Case
Single-station visual quality control on the MVTec AD dataset. For each part the agent decides disposition (pass / rework / reject), determines whether a defect is random or systematic, and triggers corrective workflows autonomously, escalating low-confidence cases to a human reviewer. Validated on two categories (`bottle`, `hazelnut`); per-category PatchCore models coexist and are selected by `VQC_CATEGORY`.

## Agent Architecture
- Brain: reasoning LLM (via LangGraph) for diagnosis interpretation and narrative generation.
- Planner: LangGraph state machine — detect -> gather context -> investigate -> reason -> decide -> act / escalate.
- Tools: PatchCore defect detector, MES query (reads long-term memory), non-conformance (NCR) writer, corrective-action (CAPA) ticket creator.
- Memory: working memory (per-inspection run state in the graph) plus long-term memory (SQLite MES store of part / machine / batch / operator history) enabling systematic-vs-random reasoning.
- Guardrails: confidence-threshold routing, deterministic decisions in code with judgment confined to the LLM, schema-validated outputs, and a full audit trail of every decision and its reasoning.

## Agentic Loop
Goal (inspect and resolve) -> Plan (route by state) -> Act (invoke tools) -> Observe (verify outputs, query history) -> Act (finalize or escalate).

## Stack
- Package manager: uv (exclusive; locked dependencies, reproducible builds).
- Orchestration: LangGraph. Perception: Anomalib (PatchCore) on PyTorch.
- Reasoning: an offline deterministic stub is the default (no key/GPU needed); a hosted Anthropic (Claude) provider — wired via structured outputs — or a local Ollama model plugs in via `VQC_LLM_PROVIDER` (+ `ANTHROPIC_API_KEY` in `.env`). Dispositions stay deterministic regardless of provider.
- Drift monitoring: self-owned ImageNet resnet18 backbone (decoupled from anomalib); kNN distance to a per-category training-good reference set; optional — off until the reference is built.
- Data: MVTec AD (CC BY-NC-SA 4.0, non-commercial — credited, not redistributed).
- Contracts / IO: Pydantic. Service: FastAPI + Docker. UI: Streamlit. Tracing: LangSmith (optional).
- Reproducibility: fixed random seeds and pinned dataset splits across training and evaluation.

## Project Structure
```
perception/   PatchCore training, detect() interface
agent/        LangGraph nodes, tools, state, guardrails
memory/       SQLite MES schema and seed data
service/      FastAPI app and Docker config
eval/         perception and agent validation, labeled scenario set
tests/        unit, regression, escalation, reasoning tests
ui/           Streamlit demo
```

## Dev notes (environment + gotchas)
- Use `uv run` for everything; never system `python` (system Python is 3.14, too new for torch/anomalib — uv pins 3.11).
- Run tests as `uv run python -m pytest`, NOT `uv run pytest` (a global pytest on 3.14 gets picked up and fails with no torch).
- When `.env` sets `VQC_LLM_PROVIDER=anthropic`, run the suite hermetically: `VQC_LLM_PROVIDER=stub uv run python -m pytest` (avoids a live API call in the service test).
- Perception artifacts are per-category under `artifacts/perception/<category>/`; switch with `VQC_CATEGORY`. `VQC_CORESET_SAMPLING_RATIO` (default 0.1) tunes PatchCore fit cost — `0.01` ("PatchCore-1%") trains ~10x faster at ~the same AUROC.
- anomalib's MVTec downloader 404s; `perception/prepare_data.py` fetches the active category from a Hugging Face mirror (idempotent).
- Drift artifacts live under `artifacts/drift/<category>/` (`reference.npz`, `drift_metrics.json`, `drift_separation.png`), parallel to `artifacts/perception/<category>/`. Build with `uv run python -m drift.reference` (after `perception.train`) then `uv run python -m eval.drift_eval`; switch category via `VQC_CATEGORY`. OOD good parts are escalated to a human; the feature is optional — off until the reference is built. `drift.report` reads the non-defective (good) stream from the MES.
- Windows: prefix non-ASCII-printing scripts with `PYTHONUTF8=1`; in the Bash tool `unset VIRTUAL_ENV` first to silence a stale-venv uv warning.

## Plan (end to end)
1. Perception core: train PatchCore on one category; expose detect(image) -> {is_defective, confidence, heatmap}.
2. Perception validation: report precision / recall, false-accept and false-reject rates, and AUROC against the error budget, using fixed seeds and pinned splits; show the threshold tradeoff.
3. Agent loop: implement LangGraph nodes over a seeded SQLite MES; emit structured decision / diagnosis / actions with an escalation path.
4. Agent validation: score the agent on a labeled scenario set with known-correct dispositions and diagnoses, measuring disposition accuracy and random-vs-systematic accuracy.
5. Service: expose the loop as a FastAPI endpoint; containerize with Docker.
6. Showcase: Streamlit demo — upload image, watch the investigation, view the three-part output.

## Success Criteria
- Perception: false-accept rate at or below target (e.g. 2 percent) at the chosen threshold; image-level AUROC reported against published baselines.
- Agent: disposition accuracy and random-vs-systematic accuracy at or above target (e.g. 95 percent) on the labeled scenario set.
- All escalation-boundary and regression tests passing; service builds and runs from a single uv-locked command.

## Test
- Unit tests on agent nodes and decision logic.
- Validation regression suite of known-good and known-defect images with pass / fail thresholds.
- Escalation tests at confidence boundaries.
- Agent-reasoning tests over the labeled scenario set (disposition and diagnosis correctness).

## Deploy
- Dockerized FastAPI service, uv-locked dependencies, reproducible build, configurable confidence threshold.

## Monitor
- Structured decision logs with full reasoning trace.
- False-accept / false-reject tracked continuously against the error budget.
- Input-distribution drift monitoring: IMPLEMENTED. Per-image kNN OOD gate on the pass path (non-defective parts scoring far from the training-good manifold are escalated to a human), plus an MES-backed PSI population monitor over the good stream. Validated: `bottle` AUROC 1.000 / `hazelnut` AUROC 0.997; calibration false-alarm ~5% (LOO). See `drift/` and `eval/drift_eval.py`.

## Slide / Demo
One-screen summary: problem, architecture diagram, perception and agent validation results versus error budget, live inspection demo.

## I/O Contract
- Input: part image plus part identifier (used to resolve machine / batch / operator from the MES).
- Output: Decision (pass / rework / reject + confidence), Diagnosis (defect type, location, random vs systematic, probable cause), Actions (NCR, CAPA, machine flag), plus a plain-language summary.

## Limitations and Future Work
- The MES is synthetic-but-realistic; the query interface is designed for real MES integration, which is the path to production.
- Vision accuracy degrades under lighting, camera, or part-variant drift; input-distribution drift monitoring is now implemented (image-level OOD detection via kNN to the training-good manifold), escalating suspect images to a human. Continuous retraining when drift is confirmed remains the production path to restoring validated accuracy.
- Single-station scope; per-category models are supported and demonstrated on `bottle` + `hazelnut`, but multi-station orchestration and logical-anomaly handling (MVTec LOCO AD) remain natural extensions.
