# Visual Quality Control Agent

Autonomous industrial inspection agent that detects part defects, diagnoses cause, and acts — going beyond detect-and-report into a goal-driven plan / act / observe / act loop, with both perception and reasoning held to defined accuracy budgets.

## Use Case
Single-station visual quality control on the MVTec AD dataset. For each part the agent decides disposition (pass / rework / reject), determines whether a defect is random or systematic, and triggers corrective workflows autonomously, escalating low-confidence cases to a human reviewer.

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
- Reasoning: hosted frontier LLM for development, with a local Ollama-served model (e.g. Llama 3.1) as an offline, key-free fallback so the repo runs without API access.
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
- Input-distribution drift check against the training set (extension; flagged as future work if scope-limited).

## Slide / Demo
One-screen summary: problem, architecture diagram, perception and agent validation results versus error budget, live inspection demo.

## I/O Contract
- Input: part image plus part identifier (used to resolve machine / batch / operator from the MES).
- Output: Decision (pass / rework / reject + confidence), Diagnosis (defect type, location, random vs systematic, probable cause), Actions (NCR, CAPA, machine flag), plus a plain-language summary.

## Limitations and Future Work
- The MES is synthetic-but-realistic; the query interface is designed for real MES integration, which is the path to production.
- Vision accuracy degrades under lighting, camera, or part-variant drift; continuous input monitoring and periodic retraining are required in a live deployment.
- Single-station, single-category scope by design; multi-station orchestration and logical-anomaly handling (MVTec LOCO AD) are natural extensions.
