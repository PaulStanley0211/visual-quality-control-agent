# Input-Distribution Drift Monitor — Design

**Date:** 2026-06-01
**Status:** Approved design — ready for implementation planning
**Project:** Visual Quality Control Agent
**Author:** Paul (with Claude)

---

## 1. Problem & motivation

The PatchCore detector's validated accuracy (image AUROC ~0.999, FAR ≤ 2%) holds **only while
incoming images resemble the training distribution**. The day a bulb is swapped, the camera is
nudged, the lens or part variant changes, the images "drift" — and the detector keeps emitting
confident-looking anomaly scores that are no longer trustworthy. This is the silent failure mode
called out in the project's own CLAUDE.md ("Monitor" section and "Limitations": *vision accuracy
degrades under lighting/camera/part-variant drift; continuous input monitoring is required*).

The drift monitor is the smoke detector for exactly this: it watches whether incoming images still
resemble the training-good distribution and (a) **escalates** individual out-of-distribution (OOD)
parts to a human because their disposition is untrustworthy, and (b) **rolls up** a population-level
drift signal over the live inspection stream so the line can be investigated / retrained before
silent false-accepts accumulate.

## 2. Scope (decisions locked during brainstorming)

- **Primary purpose:** *Both, per-image first.* A per-image OOD gate is the engine; the
  population-level monitor is an aggregation roll-up of the same per-image scores.
- **OOD behavior:** an OOD image **escalates to a human** (holds automated actions), as a new
  escalation trigger alongside the existing low-confidence / unknown-severity triggers. The drift
  read is **annotated on every inspection** regardless of escalation. One calibrated decision
  threshold; graded *visibility* without a second calibrated boundary. (A true two-tier hold is an
  explicit non-goal / YAGNI follow-up.)
- **Population monitor feed:** **reuse the MES** — store each inspection's drift score in SQLite,
  compute windowed drift on demand, mirroring how `recent_defect_rate` already works.
- **Scoring method:** **Method B — k-NN distance to a reference embedding set** (PatchCore's own
  kNN-to-coreset idea lifted to the image level). Chosen over Mahalanobis (covariance estimation is
  fragile on a few hundred high-dim samples) and over reusing the anomaly-score distribution (cannot
  separate "defect" from "OOD" per image, which is the heart of the gate).

### Non-goals

- No two-tier (warn-vs-escalate) drift hold — single threshold; annotation provides graded
  visibility.
- No standalone folder-batch drift tool — MES-backed monitor only.
- No retraining automation — the monitor *recommends* retraining; it does not trigger it.
- No coupling to anomalib internals — drift uses its own backbone forward pass.

## 3. Architecture

A new top-level **`drift/`** package, structured parallel to `perception/` (a fit step, a
calibration/eval step, and a load-and-score interface):

```
drift/
  extractor.py    # image -> embedding vector (ImageNet resnet18 backbone, global-avg-pooled)
  reference.py    # build_reference(): fit the per-category reference set from training-good images
  monitor.py      # DriftMonitor.score(image) -> DriftResult   (single interface, like detector.detect)
  report.py       # windowed PSI roll-up over the MES          (python -m drift.report)
eval/
  drift_eval.py   # synthesize drift, calibrate the OOD threshold -> drift_metrics.json + plot
```

### Per-category artifacts (parallel to `artifacts/perception/<category>/`)

```
artifacts/drift/<category>/
  reference.npz        # training-good embedding set + image-stat baselines  (built by reference.py)
  drift_metrics.json   # calibrated OOD threshold + PSI reference bins + separability report (drift_eval)
  drift_separation.png # clean-vs-drifted score-distribution plot
```

Switched by `VQC_CATEGORY`, exactly like the perception artifacts. `DriftMonitor` cross-checks the
artifact's `category` against `settings.category`, the same guard `detector._load_calibration` uses.

### Why a self-owned ImageNet resnet18 extractor (not the PatchCore pickle)

anomalib's `TorchInferencer` is already the *legacy* path; prying intermediate features out of its
pickled model would couple drift to anomalib's internal module names and version. Instead the
extractor runs the **same ImageNet-pretrained resnet18 backbone PatchCore itself uses**
(`settings.backbone`) as a small, self-owned, independently testable forward pass (~tens of ms on
CPU). "OOD in this feature space" still closely tracks "OOD for the detector," but drift stays
decoupled and testable.

### Data flow

```
                       ┌─────────── per-image path (the gate) ───────────┐
 image -> detect -> assess_drift -> gather_context -> investigate -> decide -> reason -> act | escalate
            |            |                                              ^
            |            | DriftResult                                  | is_ood -> escalate
            |            └── drift_score stored on the inspection ──────┘   (+ annotation always)
            |                            |
            v                            v
       DetectResult                  MES (SQLite) ──> drift/report.py ──> windowed PSI  (the monitor)
```

## 4. Per-image scoring

### 4.1 Embedding (`drift/extractor.py`)

Deterministic, one vector per image:

1. Load image → resize to `settings.image_size` (256) → ImageNet normalize.
2. Forward through a frozen, ImageNet-pretrained **resnet18**, tapping `layer2` and `layer3`
   (texture / lighting / structure — what drift perturbs).
3. **Global-average-pool** each block's feature map and **concatenate** → fixed ~384-dim vector.
4. **L2-normalize** (cosine geometry; robust to overall magnitude).

Frozen, `no_grad`, `eval()` mode → identical embedding for identical input (reproducibility + tests).

### 4.2 Reference artifact (`drift/reference.py` — the "fit" step, analogous to `perception.train`)

`build_reference()` runs the extractor over every **training-good** image for the active category and
writes `reference.npz`:

- `embeddings` — stack of training-good vectors (few hundred × ~384, < 1 MB).
- `stat_baseline` — mean/std of three interpretable image stats over the same images: **brightness**,
  **contrast** (RMS), **sharpness** (variance of Laplacian). These power the human-readable context
  ("brightness down 2.4σ"); **context only, never a decision input**.

Idempotent, CPU-fast (one forward pass per training image).

### 4.3 Score (`drift/monitor.py` — `DriftMonitor.score()`, analogous to `detector.detect()`)

```
drift_score(x) = mean Euclidean distance from e(x) to its k nearest reference embeddings
                 (k = settings.drift_k, default 5)
is_ood         = drift_score >= calibrated_threshold        # from drift_metrics.json
```

A clean part sits near the training-good cloud → small distance. A brighter/blurrier/different-variant
part lands in an empty region → large distance → OOD. At a few-hundred-vector scale the kNN search is
effectively instant on CPU. `DriftMonitor` loads the reference + threshold once and is reused per
image (stateful wrapper, like `Detector`).

### 4.4 Contract (`contracts/models.py`)

```python
class DriftResult(BaseModel):
    is_ood: bool          = Field(description="True if drift_score >= calibrated OOD threshold.")
    drift_score: float    = Field(description="Mean distance to k nearest training-good embeddings.")
    threshold: float      = Field(description="Calibrated OOD threshold (false-alarm budget on clean holdout).")
    brightness_delta: float | None = Field(default=None, description="Brightness vs training baseline, in std units.")
    contrast_delta: float | None   = Field(default=None, description="Contrast vs training baseline, in std units.")
    sharpness_delta: float | None  = Field(default=None, description="Sharpness vs training baseline, in std units.")
    note: str             = Field(description="Plain-language read, e.g. 'In-distribution' or 'OOD: brightness down 2.4σ'.")
```

`InspectionOutput` gains `drift: DriftResult | None = None` (None when drift wasn't assessed — no
image, or the feature is disabled).

## 5. Agent-loop integration

### 5.1 `assess_drift` node (`agent/nodes.py`)

Factory node (the `make_detect_node` pattern), inserted on `detect → assess_drift → gather_context`:

```python
def make_assess_drift_node(deps):
    def assess_drift(state):
        if not settings.drift_enabled or deps.get_drift_monitor() is None:
            return {}                              # feature off or no reference artifact -> drift stays None
        image_path = state.get("image_path")
        if not image_path:                         # scenario / precomputed-DetectResult path: skip cleanly
            return {"reasoning_trace": ["Drift: not assessed (no image provided)."]}
        try:
            dr = deps.get_drift_monitor().score(image_path)
        except Exception as e:                     # availability of the gate must never abort an inspection
            logger.warning("Drift scoring failed for '%s': %s", state["part_id"], e)
            return {"reasoning_trace": [f"Drift: unavailable ({e})."]}
        return {"drift": dr, "reasoning_trace": [f"Drift: {dr.note} (score {dr.drift_score:.3f}, OOD={dr.is_ood})."]}
    return assess_drift
```

`deps.get_drift_monitor()` is **lazy and memoized**, exactly like `deps.get_detector()`; returns
`None` if the reference artifact is absent (so the feature is simply off on a fresh clone until the
reference is built, just as detection is off until the model is trained). The graph wiring in
`agent/graph.py` adds the node and re-points the `detect → gather_context` edge through it.

### 5.2 Escalation wiring (`agent/nodes.py::decide`)

One added trigger reading `state.get("drift")`:

```python
drift = state.get("drift")
drift_ood = bool(drift and drift.is_ood)
escalated = decisions.should_escalate(...) or severity_unknown or drift_ood
```

Trace gains its reason (`"Escalate to human: image out-of-distribution (drift)."`). The disposition
is still **computed deterministically and recorded**; escalation only *holds the actions* for a
human — identical to existing escalation semantics. The drift note rides along on **every**
`InspectionOutput` via the `drift` field, escalated or not.

### 5.3 Error-handling philosophy (consistent with existing code)

| Failure | Behavior | Precedent |
|---|---|---|
| Reference artifact missing | Feature off; `drift=None`; `/health` reports it | detector requires trained model |
| Single image fails to score | Log WARNING, continue, `drift=None` | heatmap failure never fails detection (`detector.py:189`) |
| No image (scenario path) | Skip cleanly, no escalation | — |

**Fail-open for availability:** a broken drift monitor degrades to "no extra safety net," never to
"inspection crashes." The base disposition is unaffected. The loud `WARNING` log is the compensating
control. (Fail-closed → escalate-everything was considered and rejected as too noisy for this system.)

## 6. Population-level monitor

1. **MES schema:** add a nullable `drift_score REAL` column to the `inspections` table;
   `mes.record_inspection` writes `state["drift"].drift_score` or NULL. This is the only schema touch.
2. **`drift/report.py`** (`python -m drift.report`): pulls the last `settings.drift_window`
   (default 50) **production** (`source='qc'`) inspections' drift scores from the MES and compares
   their distribution to the **reference clean distribution** (bin edges + expected proportions stored
   in `drift_metrics.json` at calibration time) via **PSI**:

   ```
   PSI = Σ (actual% − expected%) · ln(actual% / expected%)
   ```

   Reported with standard bands — **< 0.1 stable · 0.1–0.25 moderate · > 0.25 significant** — plus
   the **fraction of OOD-flagged parts** in the window and the mean σ-deltas of the image stats.
   Output is a short plain-language report, e.g.
   *"Line drift: PSI 0.31 — SIGNIFICANT; 7/50 parts OOD; brightness trending −1.8σ → investigate /
   consider retraining."*

   Reuses the `source='qc'` vs `source='agent'` tagging so the monitor reflects the real production
   stream, consistent with `recent_defect_rate`.

## 7. Validation (`eval/drift_eval.py` — mirrors `perception_eval.py`)

Same honest methodology: seeded splits, calibrate on one split, report on a disjoint one, Wilson CI.

1. **Clean set = `test/good`** images — *disjoint by construction* from the reference (built on
   `train/good`), so distances aren't artificially deflated.
2. **Synthesize drift** by perturbing copies of the clean images across realistic failure modes:
   **brightness** (±), **contrast** (±), **gaussian blur** (focus drift), **gaussian noise** (sensor
   drift), **JPEG compression** (pipeline drift) — each at a couple of seeded severities.
3. **Calibrate** the OOD threshold on a seeded calibration split of the clean set so the
   **false-drift-alarm rate** (clean wrongly flagged OOD) ≤ `settings.drift_far_alarm_target`
   (default 0.05); **report** that rate on the disjoint clean holdout.
4. **Write `artifacts/drift/<category>/drift_metrics.json`:**
   - **separability AUROC** (clean vs drifted — threshold-free headline),
   - **per-perturbation detection rate** (so weak spots are visible, not averaged away),
   - false-alarm rate + **Wilson 95% upper bound** (small-sample honesty),
   - **PSI reference bins** (clean-holdout score histogram) for `drift/report.py`,
   - plus `drift_separation.png` (clean-vs-drifted distributions with the threshold drawn).

### Success criteria

- **Separability AUROC ≥ ~0.90** clean-vs-drifted on the synthetic suite (headline).
- **False-drift-alarm rate ≤ 5%** on the clean holdout, reported with Wilson upper bound.
- Every perturbation type detected **well above chance**; per-type rates published, not averaged.

## 8. Configuration (`config.py`, all `VQC_*`-overridable)

```python
drift_enabled: bool = True
drift_k: int = 5                      # kNN neighbors
drift_far_alarm_target: float = 0.05  # max clean-image false-OOD rate (calibration budget)
drift_window: int = 50                # population-monitor lookback
drift_psi_significant: float = 0.25   # PSI band for "significant drift"
# + drift_dir / drift_reference_path / drift_metrics_path properties, parallel to perception_dir
```

## 9. Surfacing (service + UI)

- **`GET /health`** → add `drift_enabled` and `drift_reference_present` (ops can see the gate is armed).
- **`POST /inspect`** → the `drift` block rides along in `InspectionOutput`; OOD parts return
  `escalated=true` with the drift reason in the trace.
- **`GET /drift`** (new, small) → returns `drift/report.py`'s windowed PSI report as JSON (one MES
  query); makes "the monitor" demoable.
- **Streamlit** → a drift badge on the result (✅ in-distribution / ⚠️ OOD, with score + the three
  σ-deltas) and a small "Line drift" panel that calls `/drift` to show the PSI band and % OOD.

## 10. Test plan (extends the existing 53-test suite; same categories)

- **Unit** — extractor output shape + determinism (same image → same vector); reference
  build/load round-trip; **scoring monotonicity** (a perturbed image scores strictly higher than its
  clean original); PSI math on hand-built distributions; σ-delta computation.
- **Escalation boundary** — clean image → not escalated by drift; OOD image → `escalated=True` +
  drift reason present.
- **Graceful degradation** — no-image path and missing-artifact path both leave `drift=None` and
  don't break existing flows; re-assert the `agent_eval` scenarios pass untouched.
- **Service** — `/inspect` on a drifted image returns the `drift` block and `escalated=true`;
  `/drift` returns a well-formed report; `/health` exposes the new fields.
- **Regression** — `test_drift_regression` asserts AUROC / false-alarm thresholds, **skipping if the
  reference artifact is absent**, exactly like `test_perception_regression` skips without a model.

All existing tests must stay green; drift being optional is what guarantees that.

## 11. Run commands (added to the project flow)

```bash
uv run python -m drift.reference        # build the per-category reference set (after perception.train)
uv run python -m eval.drift_eval        # synthesize drift, calibrate threshold -> drift_metrics.json + plot
uv run python -m drift.report           # windowed PSI report over the live MES stream
# per category: VQC_CATEGORY=hazelnut uv run python -m drift.reference   (etc.)
```

## 12. One-paragraph summary

A self-contained `drift/` package extracts an ImageNet-resnet18 image embedding, scores each part by
kNN distance to a per-category training-good reference set, escalates out-of-distribution parts to a
human while annotating the drift read on every inspection, and rolls the stored per-inspection scores
up into an MES-backed PSI line monitor — all validated the same honest, seeded, Wilson-CI way the
perception layer already is.
