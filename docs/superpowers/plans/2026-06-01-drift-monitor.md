# Input-Distribution Drift Monitor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an input-distribution drift monitor that scores each part image's distance from the training-good distribution, escalates out-of-distribution (OOD) parts to a human, annotates the drift read on every inspection, and rolls per-inspection scores up into an MES-backed PSI population monitor.

**Architecture:** A new self-contained `drift/` package (extractor → reference → monitor → report), parallel to `perception/`. An ImageNet-pretrained resnet18 backbone (the same one PatchCore uses, via `timm`) produces an image embedding; the per-image drift score is the mean kNN distance to a per-category training-good reference embedding set. A new `assess_drift` graph node feeds the existing escalation path and writes the score to the MES; `eval/drift_eval.py` calibrates the OOD threshold against synthesized drift exactly like `perception_eval.py`.

**Tech Stack:** Python 3.11 (via `uv`), PyTorch + `timm` (CPU), NumPy, Pydantic, LangGraph, SQLite, FastAPI, Streamlit, pytest.

**Design spec:** `docs/superpowers/specs/2026-06-01-drift-monitor-design.md`

---

## Conventions for every task

- Run everything via `uv run python -m ...` — never system Python (system Python is 3.14; the project pins 3.11).
- Run tests as `uv run python -m pytest` (NOT `uv run pytest`).
- In the Bash tool, prefix with `unset VIRTUAL_ENV;` to silence the stale-venv uv warning, and set `VQC_LLM_PROVIDER=stub` so no test makes a live API call. Example:
  `unset VIRTUAL_ENV; VQC_LLM_PROVIDER=stub uv run python -m pytest tests/test_drift_scoring.py -v`
- On Windows PowerShell the env-var form is `$env:VQC_LLM_PROVIDER='stub'; uv run python -m pytest ...`.
- Commit after each task with the shown message. End every commit message body with:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
- Work happens on the existing `drift-monitor` branch.

---

## File Structure

**New files**

| Path | Responsibility |
|---|---|
| `drift/__init__.py` | Package marker (empty). |
| `drift/scoring.py` | Pure math: `knn_distance()`, `population_stability_index()`. No torch. |
| `drift/stats.py` | Cheap interpretable image stats: brightness, contrast, sharpness. NumPy/PIL only. |
| `drift/extractor.py` | `EmbeddingExtractor` — image → L2-normalized ~384-dim embedding via `timm` resnet18 (`layer2`+`layer3`, GAP'd). |
| `drift/reference.py` | `Reference` dataclass, `build_reference()` (fit over training-good), `load_reference()`. `__main__` builds it. |
| `drift/monitor.py` | `DriftMonitor.score(image) -> DriftResult` — loads reference + calibrated threshold, scores per image. |
| `drift/report.py` | `population_report()` — windowed PSI + %OOD over the MES. `__main__` prints it. |
| `eval/drift_eval.py` | Synthesize drift, calibrate the OOD threshold, write `drift_metrics.json` + `drift_separation.png`. |
| `tests/test_drift_scoring.py` | Unit tests for `scoring.py` and `stats.py`. |
| `tests/test_drift_extractor.py` | Extractor shape + determinism (skips if backbone weights unavailable). |
| `tests/test_drift_reference.py` | `Reference` save/load round-trip. |
| `tests/test_drift_monitor.py` | `DriftMonitor.score` with injected reference + fake extractor (hermetic). |
| `tests/test_drift_escalation.py` | Graph integration: OOD → escalated; clean → not; no-image → drift None. |
| `tests/test_drift_report.py` | `population_report` PSI over a temp MES. |
| `tests/test_drift_regression.py` | Asserts `drift_metrics.json` AUROC/alarm budgets; skips if artifact absent. |

**Modified files**

| Path | Change |
|---|---|
| `config.py` | Add `drift_*` settings + `drift_dir`/`drift_reference_path`/`drift_metrics_path` properties. |
| `contracts/models.py` | Add `DriftResult`; add `drift` field to `InspectionOutput`. |
| `memory/schema.sql` | Add `drift_score REAL` column to `inspections`. |
| `memory/mes.py` | `_migrate()` adds the column to existing DBs; `record_inspection` writes `drift_score`. |
| `memory/seed.py` | `ensure_seeded()` migrates an already-seeded DB. |
| `agent/state.py` | `InspectionState.drift`; `AgentDeps.get_drift_monitor()`. |
| `agent/nodes.py` | `make_assess_drift_node`; `decide` treats OOD as an escalation trigger; `_record` writes drift_score. |
| `agent/graph.py` | Add `assess_drift` node, rewire `detect → assess_drift → gather_context`. |
| `service/app.py` | `/health` drift fields; new `GET /drift`. |
| `ui/streamlit_app.py` | Drift badge + "Line drift" panel. |
| `pyproject.toml` | Add `timm` dependency + `drift` pytest marker. |

---

## Task 1: Config — drift settings and artifact paths

**Files:**
- Modify: `config.py`
- Test: `tests/test_drift_scoring.py` (config assertions live at top; created here)

- [ ] **Step 1: Write the failing test**

Create `tests/test_drift_scoring.py` with just the config check for now:

```python
"""Unit tests for the drift subsystem's pure logic (config, scoring math, image stats)."""
from __future__ import annotations

from config import settings


def test_drift_config_defaults_and_paths():
    assert settings.drift_enabled is True
    assert settings.drift_k == 5
    assert settings.drift_far_alarm_target == 0.05
    assert settings.drift_window == 50
    assert settings.drift_psi_significant == 0.25
    # Per-category artifact layout mirrors perception_dir.
    assert settings.drift_dir == settings.artifacts_dir / "drift" / settings.category
    assert settings.drift_reference_path == settings.drift_dir / "reference.npz"
    assert settings.drift_metrics_path == settings.drift_dir / "drift_metrics.json"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `unset VIRTUAL_ENV; uv run python -m pytest tests/test_drift_scoring.py -v`
Expected: FAIL — `AttributeError: 'Settings' object has no attribute 'drift_enabled'`.

- [ ] **Step 3: Add the settings + properties**

In `config.py`, add these fields inside `class Settings` (after the `# --- service ---` block, before `# --- reproducibility ---`):

```python
    # --- input-distribution drift monitor ---
    drift_enabled: bool = True            # master switch; off => drift never assessed
    drift_k: int = 5                      # kNN neighbors for the per-image drift score
    drift_far_alarm_target: float = 0.05  # max clean-image false-OOD rate (threshold calibration budget)
    drift_window: int = 50                # population-monitor lookback (recent inspections with a drift score)
    drift_psi_significant: float = 0.25   # PSI at/above this => "significant" line drift
```

And add these properties after `metrics_path` (near the other `@property` defs):

```python
    @property
    def drift_dir(self) -> Path:
        """Per-category drift artifacts (reference set + calibration), parallel to perception_dir."""
        return self.artifacts_dir / "drift" / self.category

    @property
    def drift_reference_path(self) -> Path:
        return self.drift_dir / "reference.npz"

    @property
    def drift_metrics_path(self) -> Path:
        return self.drift_dir / "drift_metrics.json"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `unset VIRTUAL_ENV; uv run python -m pytest tests/test_drift_scoring.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add config.py tests/test_drift_scoring.py
git commit -m "feat(drift): add drift config settings and per-category artifact paths"
```

---

## Task 2: Contracts — `DriftResult` and `InspectionOutput.drift`

**Files:**
- Modify: `contracts/models.py`
- Test: `tests/test_drift_scoring.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_drift_scoring.py`:

```python
from contracts.models import DriftResult, InspectionOutput, Decision, Disposition


def test_drift_result_contract():
    dr = DriftResult(is_ood=True, drift_score=1.23, threshold=0.8, note="OOD: brightness down 2.4σ")
    assert dr.is_ood is True
    assert dr.brightness_delta is None  # optional context defaults to None
    assert dr.note.startswith("OOD")


def test_inspection_output_drift_defaults_none():
    out = InspectionOutput(
        part_id="P1",
        decision=Decision(disposition=Disposition.PASS, confidence=0.9),
        summary="ok",
    )
    assert out.drift is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `unset VIRTUAL_ENV; uv run python -m pytest tests/test_drift_scoring.py -v`
Expected: FAIL — `ImportError: cannot import name 'DriftResult'`.

- [ ] **Step 3: Add the contract**

In `contracts/models.py`, add after the `DetectResult` class:

```python
class DriftResult(BaseModel):
    """Output of the drift monitor for a single part image (None when drift is not assessed)."""

    is_ood: bool = Field(description="True if drift_score >= the calibrated OOD threshold.")
    drift_score: float = Field(description="Mean distance from the image embedding to its k nearest training-good embeddings.")
    threshold: float = Field(description="Calibrated OOD threshold (clean-image false-alarm budget on holdout).")
    brightness_delta: float | None = Field(default=None, description="Brightness vs training baseline, in std units.")
    contrast_delta: float | None = Field(default=None, description="Contrast vs training baseline, in std units.")
    sharpness_delta: float | None = Field(default=None, description="Sharpness vs training baseline, in std units.")
    note: str = Field(description="Plain-language read, e.g. 'In-distribution' or 'OOD: brightness down 2.4σ'.")
```

And in `InspectionOutput`, add this field (after `heatmap_path`):

```python
    drift: DriftResult | None = Field(default=None, description="Input-distribution drift assessment for this image, if assessed.")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `unset VIRTUAL_ENV; uv run python -m pytest tests/test_drift_scoring.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add contracts/models.py tests/test_drift_scoring.py
git commit -m "feat(drift): add DriftResult contract and InspectionOutput.drift field"
```

---

## Task 3: `drift/scoring.py` — pure math (kNN distance + PSI)

**Files:**
- Create: `drift/__init__.py`, `drift/scoring.py`
- Test: `tests/test_drift_scoring.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_drift_scoring.py`:

```python
import numpy as np

from drift.scoring import knn_distance, population_stability_index


def test_knn_distance_zero_on_exact_match():
    ref = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    # Query equals a reference point; with k=1 the nearest distance is 0.
    assert knn_distance(np.array([0.0, 0.0]), ref, k=1) == 0.0


def test_knn_distance_mean_of_k_nearest():
    ref = np.array([[0.0, 0.0], [3.0, 0.0], [4.0, 0.0]])
    # Distances from [0,0]: 0, 3, 4. Mean of 2 nearest = (0+3)/2 = 1.5.
    assert knn_distance(np.array([0.0, 0.0]), ref, k=2) == 1.5


def test_knn_distance_k_clamped_to_reference_size():
    ref = np.array([[0.0, 0.0], [2.0, 0.0]])
    # k larger than n uses all points: mean(0, 2) = 1.0.
    assert knn_distance(np.array([0.0, 0.0]), ref, k=10) == 1.0


def test_psi_zero_for_identical_distributions():
    p = [0.25, 0.25, 0.25, 0.25]
    assert population_stability_index(p, p) == 0.0


def test_psi_positive_and_grows_with_shift():
    expected = [0.4, 0.3, 0.2, 0.1]
    mild = [0.35, 0.3, 0.2, 0.15]
    severe = [0.1, 0.2, 0.3, 0.4]
    psi_mild = population_stability_index(expected, mild)
    psi_severe = population_stability_index(expected, severe)
    assert psi_mild > 0
    assert psi_severe > psi_mild
```

- [ ] **Step 2: Run test to verify it fails**

Run: `unset VIRTUAL_ENV; uv run python -m pytest tests/test_drift_scoring.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'drift'`.

- [ ] **Step 3: Create the package and scoring math**

Create `drift/__init__.py` (empty):

```python
```

Create `drift/scoring.py`:

```python
"""Pure scoring math for the drift monitor — no torch, fully unit-testable.

- ``knn_distance``: the per-image drift score (mean distance to the k nearest training-good
  embeddings). This is PatchCore's kNN-to-coreset idea at the image level.
- ``population_stability_index``: the windowed drift metric for the population monitor.
"""
from __future__ import annotations

import numpy as np


def knn_distance(embedding: np.ndarray, reference: np.ndarray, k: int) -> float:
    """Mean Euclidean distance from ``embedding`` to its ``k`` nearest rows of ``reference``.

    ``reference`` is the (n, d) stack of training-good embeddings. ``k`` is clamped to n so a
    tiny reference set never errors. Small distance => the image looks like the training-good
    cloud; large distance => out-of-distribution.
    """
    if reference.ndim != 2:
        raise ValueError(f"reference must be 2-D (n, d); got shape {reference.shape}")
    dists = np.linalg.norm(reference - embedding, axis=1)
    k = max(1, min(k, dists.shape[0]))
    nearest = np.partition(dists, k - 1)[:k]
    return float(np.mean(nearest))


def population_stability_index(expected: np.ndarray, actual: np.ndarray, eps: float = 1e-6) -> float:
    """Population Stability Index between two binned proportion vectors over the same bins.

    PSI = Σ (actual − expected) · ln(actual / expected). Both inputs are proportions (each
    summing to ~1) over identical bins. A small epsilon floor avoids div-by-zero / log(0) for
    empty bins. Standard interpretation: <0.1 stable, 0.1–0.25 moderate, >0.25 significant.
    """
    e = np.clip(np.asarray(expected, dtype=float), eps, None)
    a = np.clip(np.asarray(actual, dtype=float), eps, None)
    if e.shape != a.shape:
        raise ValueError(f"expected and actual must share shape; got {e.shape} vs {a.shape}")
    return float(np.sum((a - e) * np.log(a / e)))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `unset VIRTUAL_ENV; uv run python -m pytest tests/test_drift_scoring.py -v`
Expected: PASS (all scoring tests green).

- [ ] **Step 5: Commit**

```bash
git add drift/__init__.py drift/scoring.py tests/test_drift_scoring.py
git commit -m "feat(drift): add pure scoring math (kNN distance + PSI)"
```

---

## Task 4: `drift/stats.py` — interpretable image statistics

**Files:**
- Create: `drift/stats.py`
- Test: `tests/test_drift_scoring.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_drift_scoring.py`:

```python
from PIL import Image

from drift.stats import image_stats


def test_image_stats_keys_and_brightness_ordering():
    dark = Image.new("RGB", (32, 32), (20, 20, 20))
    bright = Image.new("RGB", (32, 32), (200, 200, 200))
    sd, sb = image_stats(dark), image_stats(bright)
    assert set(sd) == {"brightness", "contrast", "sharpness"}
    assert sb["brightness"] > sd["brightness"]


def test_image_stats_contrast_and_sharpness_flat_image_is_zero():
    flat = Image.new("RGB", (32, 32), (128, 128, 128))
    s = image_stats(flat)
    assert s["contrast"] == 0.0      # uniform image has no spread
    assert s["sharpness"] == 0.0     # uniform image has no edges
```

- [ ] **Step 2: Run test to verify it fails**

Run: `unset VIRTUAL_ENV; uv run python -m pytest tests/test_drift_scoring.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'drift.stats'`.

- [ ] **Step 3: Implement the stats**

Create `drift/stats.py`:

```python
"""Cheap, interpretable image statistics for drift *context* (never a decision input).

brightness = mean luminance, contrast = luminance std (RMS contrast), sharpness = variance of
the Laplacian (low => blurry/out-of-focus). Reported as σ-deltas vs the training baseline so a
human reviewer sees, e.g., 'brightness down 2.4σ'. Pure NumPy/PIL (no OpenCV).
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def _laplacian(gray: np.ndarray) -> np.ndarray:
    """4-neighbour Laplacian via edge-padded shifts (no SciPy/OpenCV dependency)."""
    p = np.pad(gray, 1, mode="edge")
    return p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] - 4.0 * gray


def image_stats(img: Image.Image) -> dict[str, float]:
    """Return {brightness, contrast, sharpness} for a PIL image (luminance, 0–255 scale)."""
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    return {
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "sharpness": float(_laplacian(gray).var()),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `unset VIRTUAL_ENV; uv run python -m pytest tests/test_drift_scoring.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add drift/stats.py tests/test_drift_scoring.py
git commit -m "feat(drift): add interpretable image statistics (brightness/contrast/sharpness)"
```

---

## Task 5: `drift/extractor.py` — backbone embedding

**Files:**
- Create: `drift/extractor.py`
- Modify: `pyproject.toml` (add `timm` dependency)
- Test: `tests/test_drift_extractor.py`

- [ ] **Step 1: Add the `timm` dependency**

In `pyproject.toml`, add `"timm",` to the `dependencies` list (after `"scikit-learn",`). `timm` is already present transitively via anomalib; declaring it makes the drift backbone an intentional dependency. Then sync:

Run: `unset VIRTUAL_ENV; uv sync`
Expected: resolves with no significant new downloads (timm already in the lock via anomalib).

- [ ] **Step 2: Write the failing test**

Create `tests/test_drift_extractor.py`:

```python
"""EmbeddingExtractor shape + determinism. Skips if the backbone weights can't be constructed
(e.g. offline clean clone with no cached timm weights)."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image


def _extractor_or_skip():
    try:
        from drift.extractor import EmbeddingExtractor

        return EmbeddingExtractor()
    except Exception as e:  # noqa: BLE001 - any construction failure => environment can't run this test
        pytest.skip(f"backbone unavailable: {e}")


def test_embedding_shape_and_l2_normalized():
    ext = _extractor_or_skip()
    img = Image.new("RGB", (256, 256), (123, 116, 100))
    emb = ext.embed(img)
    assert emb.shape == (384,)                       # layer2 (128) + layer3 (256), GAP'd
    assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-4)  # L2-normalized


def test_embedding_is_deterministic():
    ext = _extractor_or_skip()
    img = Image.new("RGB", (256, 256), (90, 90, 90))
    np.testing.assert_allclose(ext.embed(img), ext.embed(img), atol=1e-6)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `unset VIRTUAL_ENV; uv run python -m pytest tests/test_drift_extractor.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'drift.extractor'` (not a skip — the import inside `_extractor_or_skip` raises ImportError, which is caught and turned into a skip; so expect SKIPPED here). If SKIPPED, that still confirms the module is missing. Proceed to implement; after Step 5 it must PASS (not skip) on the dev machine where weights are cached.

- [ ] **Step 4: Implement the extractor**

Create `drift/extractor.py`:

```python
"""Image -> embedding for drift scoring.

Uses the same ImageNet-pretrained resnet18 backbone family PatchCore itself uses (via ``timm``),
tapping ``layer2`` + ``layer3`` (the layers anomalib's PatchCore uses), global-average-pooling each
and concatenating into a fixed ~384-dim vector, L2-normalized. Self-owned so drift stays decoupled
from anomalib's (legacy) TorchInferencer internals. Frozen + eval mode => deterministic.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from config import settings

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class EmbeddingExtractor:
    """Stateful backbone wrapper; construct once, call ``embed()`` per image (CPU)."""

    def __init__(self, backbone: str | None = None, image_size: int | None = None):
        import timm

        name = backbone or settings.backbone
        size = image_size or settings.image_size
        # features_only with out_indices (2, 3) => [layer2, layer3] feature maps for a resnet.
        self.model = timm.create_model(name, pretrained=True, features_only=True, out_indices=(2, 3))
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])

    @torch.no_grad()
    def embed(self, img: Image.Image) -> np.ndarray:
        """Return the L2-normalized embedding (float32, shape (d,)) for a PIL image."""
        x = self.transform(img.convert("RGB")).unsqueeze(0)         # (1, 3, H, W)
        feats = self.model(x)                                       # list of (1, C, h, w)
        pooled = [f.mean(dim=(2, 3)).squeeze(0) for f in feats]     # global average pool
        emb = torch.cat(pooled).numpy().astype(np.float32)
        norm = float(np.linalg.norm(emb))
        return emb / norm if norm > 0 else emb
```

- [ ] **Step 5: Run test to verify it passes**

Run: `unset VIRTUAL_ENV; uv run python -m pytest tests/test_drift_extractor.py -v`
Expected: PASS on the dev machine (weights cached from PatchCore training). SKIPPED is acceptable only where backbone weights are genuinely unavailable.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock drift/extractor.py tests/test_drift_extractor.py
git commit -m "feat(drift): add resnet18 backbone embedding extractor"
```

---

## Task 6: `drift/reference.py` — build + load the reference set

**Files:**
- Create: `drift/reference.py`
- Test: `tests/test_drift_reference.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_drift_reference.py`:

```python
"""Reference save/load round-trip (no backbone needed — we hand-build a Reference)."""
from __future__ import annotations

import numpy as np

from config import settings
from drift.reference import Reference, load_reference, save_reference


def test_reference_round_trip(tmp_path):
    ref = Reference(
        embeddings=np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
        stat_mean={"brightness": 100.0, "contrast": 30.0, "sharpness": 500.0},
        stat_std={"brightness": 10.0, "contrast": 5.0, "sharpness": 50.0},
        category=settings.category,
    )
    path = tmp_path / "reference.npz"
    save_reference(ref, path)
    loaded = load_reference(path)

    np.testing.assert_allclose(loaded.embeddings, ref.embeddings)
    assert loaded.stat_mean == ref.stat_mean
    assert loaded.stat_std == ref.stat_std
    assert loaded.category == ref.category
```

- [ ] **Step 2: Run test to verify it fails**

Run: `unset VIRTUAL_ENV; uv run python -m pytest tests/test_drift_reference.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'drift.reference'`.

- [ ] **Step 3: Implement reference build/load**

Create `drift/reference.py`:

```python
"""Build and load the per-category training-good reference embedding set (the drift "fit" step).

``build_reference()`` runs the extractor over every training-good image for the active category and
persists the embedding stack plus the brightness/contrast/sharpness baselines. Analogous to
``perception.train`` (a one-pass fit), idempotent, CPU-fast.

Run:  uv run python -m drift.reference
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from config import settings
from drift.stats import image_stats

_STAT_KEYS = ("brightness", "contrast", "sharpness")


@dataclass
class Reference:
    """The drift reference: training-good embeddings + image-stat baselines for one category."""

    embeddings: np.ndarray          # (n, d) float32, L2-normalized
    stat_mean: dict[str, float]
    stat_std: dict[str, float]
    category: str


def save_reference(ref: Reference, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        embeddings=ref.embeddings.astype(np.float32),
        stat_keys=np.array(_STAT_KEYS),
        stat_mean=np.array([ref.stat_mean[k] for k in _STAT_KEYS], dtype=np.float64),
        stat_std=np.array([ref.stat_std[k] for k in _STAT_KEYS], dtype=np.float64),
        category=np.array(ref.category),
    )


def load_reference(path: Path | None = None) -> Reference:
    path = path or settings.drift_reference_path
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Drift reference not found at {path}. Run `uv run python -m drift.reference` "
            "(after `perception.train`) to build it for the active category."
        )
    data = np.load(path, allow_pickle=False)
    keys = [str(k) for k in data["stat_keys"]]
    return Reference(
        embeddings=data["embeddings"].astype(np.float32),
        stat_mean=dict(zip(keys, (float(v) for v in data["stat_mean"]))),
        stat_std=dict(zip(keys, (float(v) for v in data["stat_std"]))),
        category=str(data["category"]),
    )


def _train_good_dir() -> Path:
    return settings.data_root / "MVTecAD" / settings.category / "train" / "good"


def build_reference() -> Reference:
    """Fit the reference set from the category's training-good images and persist it."""
    from drift.extractor import EmbeddingExtractor

    good_dir = _train_good_dir()
    images = sorted(good_dir.glob("*.png"))
    if not images:
        raise FileNotFoundError(
            f"No training-good images at {good_dir}. Run `perception.train` (which fetches the category) first."
        )

    extractor = EmbeddingExtractor()
    embeddings = np.empty((len(images), 384), dtype=np.float32)
    stats_accum = {k: [] for k in _STAT_KEYS}
    print(f"[drift.reference] Embedding {len(images)} training-good images for '{settings.category}' (CPU)...")
    for i, p in enumerate(images):
        img = Image.open(p).convert("RGB")
        embeddings[i] = extractor.embed(img)
        s = image_stats(img)
        for k in _STAT_KEYS:
            stats_accum[k].append(s[k])

    stat_mean = {k: float(np.mean(v)) for k, v in stats_accum.items()}
    stat_std = {k: float(np.std(v)) for k, v in stats_accum.items()}
    ref = Reference(embeddings=embeddings, stat_mean=stat_mean, stat_std=stat_std, category=settings.category)
    save_reference(ref, settings.drift_reference_path)
    print(f"[drift.reference] Saved reference ({len(images)} embeddings) to {settings.drift_reference_path}")
    return ref


if __name__ == "__main__":
    build_reference()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `unset VIRTUAL_ENV; uv run python -m pytest tests/test_drift_reference.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add drift/reference.py tests/test_drift_reference.py
git commit -m "feat(drift): build/load per-category training-good reference set"
```

---

## Task 7: `drift/monitor.py` — `DriftMonitor.score`

**Files:**
- Create: `drift/monitor.py`
- Test: `tests/test_drift_monitor.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_drift_monitor.py`:

```python
"""DriftMonitor.score with an injected reference + fake extractor (no backbone needed)."""
from __future__ import annotations

import numpy as np
from PIL import Image

from config import settings
from drift.monitor import DriftMonitor
from drift.reference import Reference


class _FakeExtractor:
    def __init__(self, vec):
        self._vec = np.asarray(vec, dtype=np.float32)

    def embed(self, img):
        return self._vec


def _reference():
    # Five identical reference points at the origin in 4-D.
    return Reference(
        embeddings=np.zeros((5, 4), dtype=np.float32),
        stat_mean={"brightness": 128.0, "contrast": 30.0, "sharpness": 500.0},
        stat_std={"brightness": 10.0, "contrast": 5.0, "sharpness": 50.0},
        category=settings.category,
    )


def _img():
    return Image.new("RGB", (32, 32), (128, 128, 128))


def test_far_embedding_is_ood():
    m = DriftMonitor(reference=_reference(), threshold=0.5, extractor=_FakeExtractor([1.0, 0.0, 0.0, 0.0]))
    res = m.score(_img())
    assert res.is_ood is True              # distance 1.0 >= threshold 0.5
    assert res.drift_score == 1.0
    assert res.note.startswith("OOD")


def test_near_embedding_in_distribution():
    m = DriftMonitor(reference=_reference(), threshold=2.0, extractor=_FakeExtractor([1.0, 0.0, 0.0, 0.0]))
    res = m.score(_img())
    assert res.is_ood is False             # distance 1.0 < threshold 2.0
    assert res.note == "In-distribution"


def test_stat_deltas_reported_in_sigma_units():
    m = DriftMonitor(reference=_reference(), threshold=2.0, extractor=_FakeExtractor([0.0, 0.0, 0.0, 0.0]))
    res = m.score(_img())  # flat 128 image: brightness 128 == baseline mean => 0σ
    assert res.brightness_delta == 0.0
    assert res.contrast_delta is not None and res.sharpness_delta is not None


def test_category_mismatch_rejected():
    ref = _reference()
    ref.category = "not-the-active-category"
    try:
        DriftMonitor(reference=ref, threshold=1.0, extractor=_FakeExtractor([0, 0, 0, 0]))
        assert False, "expected category mismatch to raise"
    except ValueError:
        pass
```

- [ ] **Step 2: Run test to verify it fails**

Run: `unset VIRTUAL_ENV; uv run python -m pytest tests/test_drift_monitor.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'drift.monitor'`.

- [ ] **Step 3: Implement the monitor**

Create `drift/monitor.py`:

```python
"""Per-image drift scoring: DriftMonitor.score(image) -> DriftResult.

Loads the per-category training-good reference set and the calibrated OOD threshold, then scores
each image by mean kNN distance to the reference (PatchCore's idea at the image level). The raw
disposition is unaffected — an OOD result is consumed by the agent's escalation logic. Stateful
wrapper, load once and reuse, exactly like ``perception.detector.Detector``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from config import settings
from contracts.models import DriftResult
from drift.reference import Reference, load_reference
from drift.scoring import knn_distance
from drift.stats import image_stats

_STAT_LABELS = {"brightness": "brightness", "contrast": "contrast", "sharpness": "sharpness"}


def _load_threshold() -> float:
    """Read the calibrated OOD threshold from drift_metrics.json (written by eval.drift_eval)."""
    path = settings.drift_metrics_path
    if not path.exists():
        raise FileNotFoundError(
            f"Drift calibration not found at {path}. Run `uv run python -m eval.drift_eval` after building the reference."
        )
    data = json.loads(path.read_text())
    if data.get("category") != settings.category:
        raise ValueError(
            f"Drift calibration is for '{data.get('category')}' but settings.category is '{settings.category}'. "
            "Re-run eval.drift_eval for the active category."
        )
    t = float(data["operating_threshold"])
    if not math.isfinite(t):
        raise ValueError(f"Drift calibration has a non-finite threshold ({t}). Re-run eval.drift_eval.")
    return t


def _describe(is_ood: bool, deltas: dict[str, float]) -> str:
    """Plain-language read; for OOD, name the most extreme image-stat delta as a hint."""
    if not is_ood:
        return "In-distribution"
    key = max(deltas, key=lambda k: abs(deltas[k]))
    direction = "up" if deltas[key] >= 0 else "down"
    return f"OOD: {_STAT_LABELS[key]} {direction} {abs(deltas[key]):.1f}σ"


class DriftMonitor:
    """Load the reference + threshold once; score each image. Inject deps for testing."""

    def __init__(self, reference: Reference | None = None, threshold: float | None = None, extractor=None):
        self.reference = reference or load_reference(settings.drift_reference_path)
        if self.reference.category != settings.category:
            raise ValueError(
                f"Drift reference is for '{self.reference.category}' but settings.category is "
                f"'{settings.category}'. Re-build the reference for the active category."
            )
        self.threshold = _load_threshold() if threshold is None else float(threshold)
        self.k = settings.drift_k
        self._extractor = extractor  # lazily built if not injected

    @property
    def extractor(self):
        if self._extractor is None:
            from drift.extractor import EmbeddingExtractor

            self._extractor = EmbeddingExtractor()
        return self._extractor

    def score(self, image: str | Path | Image.Image) -> DriftResult:
        try:
            img = Image.open(image).convert("RGB") if isinstance(image, (str, Path)) else image.convert("RGB")
        except (OSError, UnidentifiedImageError) as e:
            raise ValueError(f"Could not read image for drift scoring ({image!r}): {e}") from e

        emb = self.extractor.embed(img)
        dist = knn_distance(emb, self.reference.embeddings, self.k)
        is_ood = bool(dist >= self.threshold)

        stats = image_stats(img)
        deltas = {
            k: (stats[k] - self.reference.stat_mean[k]) / max(self.reference.stat_std[k], 1e-6)
            for k in stats
        }
        return DriftResult(
            is_ood=is_ood,
            drift_score=round(dist, 6),
            threshold=round(self.threshold, 6),
            brightness_delta=round(deltas["brightness"], 3),
            contrast_delta=round(deltas["contrast"], 3),
            sharpness_delta=round(deltas["sharpness"], 3),
            note=_describe(is_ood, deltas),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `unset VIRTUAL_ENV; uv run python -m pytest tests/test_drift_monitor.py -v`
Expected: PASS (all four tests).

- [ ] **Step 5: Commit**

```bash
git add drift/monitor.py tests/test_drift_monitor.py
git commit -m "feat(drift): add DriftMonitor.score producing DriftResult"
```

---

## Task 8: MES — drift_score column, migration, and write path

**Files:**
- Modify: `memory/schema.sql`, `memory/mes.py`, `memory/seed.py`
- Test: `tests/test_drift_report.py` (created here; reused in Task 11)

- [ ] **Step 1: Write the failing test**

Create `tests/test_drift_report.py`:

```python
"""MES drift_score write path + (Task 11) population report."""
from __future__ import annotations

from memory import mes
from memory import seed as seed_module


def test_record_inspection_persists_drift_score(tmp_path):
    db = tmp_path / "mes.db"
    seed_module.seed(db, verbose=False)
    conn = mes.connect(db)
    try:
        mes.record_inspection(
            conn,
            part_id="SCN-GOOD-1",
            is_defective=False,
            confidence=0.9,
            anomaly_score=0.1,
            defect_type=None,
            disposition="pass",
            fault_pattern=None,
            escalated=False,
            reasoning="ok",
            actions={},
            drift_score=1.234,
            source="agent",
        )
        row = conn.execute(
            "SELECT drift_score FROM inspections WHERE part_id='SCN-GOOD-1' ORDER BY inspection_id DESC LIMIT 1"
        ).fetchone()
        assert row["drift_score"] == 1.234
    finally:
        conn.close()


def test_migrate_adds_drift_score_to_legacy_table(tmp_path):
    db = tmp_path / "legacy.db"
    conn = mes.connect(db)
    try:
        # Simulate a pre-drift inspections table (no drift_score column).
        conn.execute(
            "CREATE TABLE inspections (inspection_id INTEGER PRIMARY KEY AUTOINCREMENT, part_id TEXT, source TEXT)"
        )
        conn.commit()
        mes._migrate(conn)
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(inspections)").fetchall()}
        assert "drift_score" in cols
    finally:
        conn.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `unset VIRTUAL_ENV; uv run python -m pytest tests/test_drift_report.py -v`
Expected: FAIL — `record_inspection() got an unexpected keyword argument 'drift_score'` (and `_migrate` missing).

- [ ] **Step 3: Update schema, migration, and write path**

In `memory/schema.sql`, add the column to the `inspections` table (after `anomaly_score REAL,`):

```sql
    drift_score   REAL,                           -- input-distribution drift score (NULL if not assessed)
```

In `memory/mes.py`, add `_migrate` and call it from `init_db`:

```python
def _migrate(conn: sqlite3.Connection) -> None:
    """Idempotent, additive migrations for databases created before a column existed."""
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(inspections)").fetchall()}
    if "drift_score" not in cols:
        conn.execute("ALTER TABLE inspections ADD COLUMN drift_score REAL")


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_PATH.read_text())
    _migrate(conn)
    conn.commit()
```

(The existing `init_db` body is replaced by the version above — it just adds the `_migrate(conn)` call before `commit`.)

Update `record_inspection` to accept and write `drift_score`. Change the signature to add `drift_score: float | None = None` (place it after `anomaly_score`), and update the INSERT:

```python
def record_inspection(
    conn: sqlite3.Connection,
    *,
    part_id: str,
    is_defective: bool,
    confidence: float | None,
    anomaly_score: float | None,
    drift_score: float | None = None,
    defect_type: str | None,
    disposition: str | None,
    fault_pattern: str | None,
    escalated: bool,
    reasoning: str | None,
    actions: dict,
    source: str = "agent",
    commit: bool = True,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO inspections
          (part_id, ts, is_defective, confidence, anomaly_score, drift_score, defect_type,
           disposition, fault_pattern, escalated, reasoning, actions_json, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            part_id, _now_iso(), int(is_defective), confidence, anomaly_score, drift_score, defect_type,
            disposition, fault_pattern, int(escalated), reasoning, json.dumps(actions), source,
        ),
    )
    if commit:
        conn.commit()
    return int(cur.lastrowid)
```

In `memory/seed.py`, make `ensure_seeded` self-heal an already-seeded legacy DB. Replace the `if n == 0: seed(...)` tail of `ensure_seeded` with:

```python
    if n == 0:
        seed(db_path, verbose=False)
    else:
        # Already seeded: ensure the schema is current (additive migrations only).
        conn = mes.connect(path)
        try:
            with conn:
                mes._migrate(conn)
        finally:
            conn.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `unset VIRTUAL_ENV; uv run python -m pytest tests/test_drift_report.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Rebuild the local MES so it has the new column**

Run: `unset VIRTUAL_ENV; uv run python -m memory.seed`
Expected: `[seed] 140 parts, 130 inspections ...` and a recent-defect-rate summary.

- [ ] **Step 6: Commit**

```bash
git add memory/schema.sql memory/mes.py memory/seed.py tests/test_drift_report.py
git commit -m "feat(drift): persist per-inspection drift_score in the MES (+ legacy migration)"
```

---

## Task 9: AgentState + AgentDeps wiring

**Files:**
- Modify: `agent/state.py`
- Test: `tests/test_drift_escalation.py` (created here; extended in Task 10)

- [ ] **Step 1: Write the failing test**

Create `tests/test_drift_escalation.py`:

```python
"""Drift escalation wiring + dependency injection."""
from __future__ import annotations

from agent.state import AgentDeps
from contracts.models import DriftResult


class _StubMonitor:
    def __init__(self, result):
        self._result = result

    def score(self, image):
        return self._result


def test_get_drift_monitor_returns_injected():
    stub = _StubMonitor(DriftResult(is_ood=True, drift_score=9.0, threshold=1.0, note="OOD: x up 9.0σ"))
    deps = AgentDeps(_drift_monitor=stub)
    assert deps.get_drift_monitor() is stub


def test_get_drift_monitor_none_when_artifact_absent(tmp_path, monkeypatch):
    # No reference artifact in a temp dir => get_drift_monitor must degrade to None, not raise.
    from config import settings

    monkeypatch.setattr(settings, "artifacts_dir", tmp_path)
    deps = AgentDeps()
    assert deps.get_drift_monitor() is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `unset VIRTUAL_ENV; uv run python -m pytest tests/test_drift_escalation.py -v`
Expected: FAIL — `TypeError: AgentDeps.__init__() got an unexpected keyword argument '_drift_monitor'`.

- [ ] **Step 3: Wire state + deps**

In `agent/state.py`, import `DriftResult`:

```python
from contracts.models import Actions, Decision, DetectResult, Diagnosis, DriftResult, InspectionOutput
```

Add a `drift` field to `InspectionState` (after `detect_result`):

```python
    drift: Optional[DriftResult]  # input-distribution drift assessment (None if not assessed)
```

Add the lazy, memoized drift-monitor accessor to `AgentDeps` (add the two fields and the method; mirrors `_detector`/`get_detector`):

```python
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
```

(Replace the existing `_detector`/`get_detector` block with the version above so the new fields and method sit alongside it.)

- [ ] **Step 4: Run test to verify it passes**

Run: `unset VIRTUAL_ENV; uv run python -m pytest tests/test_drift_escalation.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add agent/state.py tests/test_drift_escalation.py
git commit -m "feat(drift): add drift to InspectionState and lazy DriftMonitor on AgentDeps"
```

---

## Task 10: Graph node — `assess_drift` + escalation + audit write

**Files:**
- Modify: `agent/nodes.py`, `agent/graph.py`
- Test: `tests/test_drift_escalation.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_drift_escalation.py`:

```python
import pytest
from PIL import Image

from agent.graph import build_graph, run_inspection
from agent.llm import StubProvider
from contracts.models import DetectResult
from memory import mes, seed as seed_module


@pytest.fixture
def db(tmp_path):
    p = tmp_path / "mes.db"
    seed_module.seed(p, verbose=False)
    return p


def _png(tmp_path):
    p = tmp_path / "part.png"
    Image.new("RGB", (16, 16), (120, 120, 120)).save(p)
    return str(p)


def _clean_detect():
    # Confident GOOD: would NOT escalate on its own.
    return DetectResult(is_defective=False, confidence=0.95, anomaly_score=0.1, threshold=0.5, defect_area=0.0)


def test_ood_image_escalates_confident_good(db, tmp_path):
    ood = DriftResult(is_ood=True, drift_score=9.0, threshold=1.0, note="OOD: brightness down 3.0σ")
    deps = AgentDeps(db_path=db, provider=StubProvider(), _drift_monitor=_StubMonitor(ood))
    out = run_inspection(build_graph(deps), "SCN-GOOD-1", image_path=_png(tmp_path), detect_result=_clean_detect())
    assert out.drift is not None and out.drift.is_ood is True
    assert out.escalated is True                                  # drift forced the hold
    assert out.actions.ncr is False and out.actions.capa is False  # actions held
    # The drift score is persisted on the audit row.
    conn = mes.connect(db)
    try:
        row = conn.execute(
            "SELECT drift_score FROM inspections WHERE part_id='SCN-GOOD-1' ORDER BY inspection_id DESC LIMIT 1"
        ).fetchone()
        assert row["drift_score"] == 9.0
    finally:
        conn.close()


def test_in_distribution_image_does_not_escalate(db, tmp_path):
    ok = DriftResult(is_ood=False, drift_score=0.2, threshold=1.0, note="In-distribution")
    deps = AgentDeps(db_path=db, provider=StubProvider(), _drift_monitor=_StubMonitor(ok))
    out = run_inspection(build_graph(deps), "SCN-GOOD-1", image_path=_png(tmp_path), detect_result=_clean_detect())
    assert out.drift is not None and out.drift.is_ood is False
    assert out.escalated is False


def test_no_image_skips_drift_cleanly(db):
    # Scenario path: detect_result injected, no image => drift not assessed, behavior unchanged.
    deps = AgentDeps(db_path=db, provider=StubProvider(), _drift_monitor=_StubMonitor(
        DriftResult(is_ood=True, drift_score=9.0, threshold=1.0, note="OOD")
    ))
    out = run_inspection(build_graph(deps), "SCN-GOOD-1", detect_result=_clean_detect())
    assert out.drift is None
    assert out.escalated is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `unset VIRTUAL_ENV; uv run python -m pytest tests/test_drift_escalation.py -v`
Expected: FAIL — graph has no `assess_drift` node and `out.drift` is None for the OOD case (escalation not wired).

- [ ] **Step 3: Add the node, escalation trigger, and audit write**

In `agent/nodes.py`, add the new factory node (place it after `make_detect_node`):

```python
# --- assess_drift (input-distribution OOD gate) ---

def make_assess_drift_node(deps: AgentDeps):
    def assess_drift(state: InspectionState) -> dict:
        if not settings.drift_enabled:
            return {}
        monitor = deps.get_drift_monitor()
        if monitor is None:
            return {}  # feature off / artifact absent -> drift stays None
        image_path = state.get("image_path")
        if not image_path:
            return {"reasoning_trace": ["Drift: not assessed (no image provided)."]}
        try:
            dr = monitor.score(image_path)
        except Exception as e:  # noqa: BLE001 - the gate's availability must never abort an inspection
            logger.warning("Drift scoring failed for part '%s': %s", state["part_id"], e)
            return {"reasoning_trace": [f"Drift: unavailable ({e})."]}
        return {
            "drift": dr,
            "reasoning_trace": [f"Drift: {dr.note} (score {dr.drift_score:.3f}, OOD={dr.is_ood})."],
        }

    return assess_drift
```

In `agent/nodes.py::decide`, add the OOD trigger and reason. Replace the escalation computation and trace block with:

```python
    pattern_relevant = decisions.pattern_affects_disposition(dr.is_defective, severe)
    eff_diag_conf = inv["diagnosis_confidence"] if pattern_relevant else 1.0
    conf_low = decisions.should_escalate(dr.confidence, eff_diag_conf, settings.confidence_threshold)
    drift = state.get("drift")
    drift_ood = bool(drift and drift.is_ood)
    escalated = conf_low or severity_unknown or drift_ood
    routing_conf = min(dr.confidence, eff_diag_conf)

    trace = f"Decision: {disposition.value} (routing confidence {routing_conf:.2f})."
    if escalated:
        reasons = []
        if conf_low:
            reasons.append("confidence below threshold")
        if severity_unknown:
            reasons.append("unknown severity (no anomaly extent)")
        if drift_ood:
            reasons.append("image out-of-distribution (drift)")
        trace += " Escalate to human: " + "; ".join(reasons) + "."
```

In `agent/nodes.py::_record`, persist the drift score. Add this near the top of `_record` (after `dr = state["detect_result"]`):

```python
    drift = state.get("drift")
```

and add `drift_score=drift.drift_score if drift else None,` to the `mes.record_inspection(...)` call (place it right after `anomaly_score=dr.anomaly_score,`).

In `agent/graph.py`, register the node and rewire the edge. Add after the `g.add_node("detect", ...)` line:

```python
    g.add_node("assess_drift", nodes.make_assess_drift_node(deps))
```

and replace the edge `g.add_edge("detect", "gather_context")` with:

```python
    g.add_edge("detect", "assess_drift")
    g.add_edge("assess_drift", "gather_context")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `unset VIRTUAL_ENV; uv run python -m pytest tests/test_drift_escalation.py -v`
Expected: PASS (all five tests in the file).

- [ ] **Step 5: Run the existing agent/escalation suites to confirm no regressions**

Run: `unset VIRTUAL_ENV; VQC_LLM_PROVIDER=stub uv run python -m pytest tests/test_nodes.py tests/test_escalation.py tests/test_reasoning_scenarios.py -v`
Expected: PASS — the new node is a no-op when no image/monitor is present, so existing scenarios are unchanged.

- [ ] **Step 6: Commit**

```bash
git add agent/nodes.py agent/graph.py tests/test_drift_escalation.py
git commit -m "feat(drift): assess_drift node escalates OOD parts and records drift_score"
```

---

## Task 11: `drift/report.py` — population PSI monitor

**Files:**
- Create: `drift/report.py`
- Test: `tests/test_drift_report.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_drift_report.py`:

```python
from drift.report import population_report

_METRICS = {
    "category": "bottle",
    "operating_threshold": 1.0,
    "psi_reference": {
        # Two bins split at the threshold; reference is mostly in-distribution (low scores).
        "bin_edges": [0.0, 1.0, 100.0],
        "expected_props": [0.9, 0.1],
    },
}


def _insert(conn, part_id, score):
    conn.execute(
        """INSERT INTO inspections (part_id, ts, is_defective, drift_score, source)
           VALUES (?, '2026-06-01T00:00:00+00:00', 0, ?, 'agent')""",
        (part_id, score),
    )


def test_population_report_flags_significant_drift(tmp_path):
    db = tmp_path / "mes.db"
    seed_module.seed(db, verbose=False)
    conn = mes.connect(db)
    try:
        # All recent processed images are high-score (drifted) — far from the reference's 90/10 split.
        for i in range(20):
            _insert(conn, "SCN-GOOD-1", 5.0)
        conn.commit()
        report = population_report(conn, window=20, metrics=_METRICS)
        assert report["n"] == 20
        assert report["frac_ood"] == 1.0           # every score >= threshold 1.0
        assert report["psi"] > 0.25                 # large divergence
        assert report["band"] == "significant"
    finally:
        conn.close()


def test_population_report_empty_when_no_drift_scores(tmp_path):
    db = tmp_path / "mes.db"
    seed_module.seed(db, verbose=False)   # seed rows are 'qc' with NULL drift_score
    conn = mes.connect(db)
    try:
        report = population_report(conn, window=50, metrics=_METRICS)
        assert report["n"] == 0
        assert report["band"] == "no-data"
    finally:
        conn.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `unset VIRTUAL_ENV; uv run python -m pytest tests/test_drift_report.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'drift.report'`.

- [ ] **Step 3: Implement the report**

Create `drift/report.py`:

```python
"""Population-level drift monitor: windowed PSI + %OOD over the MES.

Reads the most recent inspections that carry a drift score (the real processed-image stream;
synthetic 'qc' seed rows have NULL drift_score and are excluded) and compares their score
distribution to the calibrated reference distribution via PSI.

Run:  uv run python -m drift.report
"""
from __future__ import annotations

import json

import numpy as np

from config import settings
from drift.scoring import population_stability_index


def _recent_scores(conn, window: int) -> list[float]:
    rows = conn.execute(
        """
        SELECT drift_score FROM inspections
        WHERE drift_score IS NOT NULL
        ORDER BY ts DESC, inspection_id DESC
        LIMIT ?
        """,
        (window,),
    ).fetchall()
    return [float(r["drift_score"]) for r in rows]


def _band(psi: float) -> str:
    if psi < 0.1:
        return "stable"
    if psi < settings.drift_psi_significant:
        return "moderate"
    return "significant"


def population_report(conn, window: int | None = None, metrics: dict | None = None) -> dict:
    """Compute the windowed drift report. ``metrics`` defaults to the on-disk drift_metrics.json."""
    window = window or settings.drift_window
    if metrics is None:
        metrics = json.loads(settings.drift_metrics_path.read_text())
    ref = metrics["psi_reference"]
    bin_edges = np.asarray(ref["bin_edges"], dtype=float)
    expected = np.asarray(ref["expected_props"], dtype=float)
    threshold = float(metrics["operating_threshold"])

    scores = _recent_scores(conn, window)
    if not scores:
        return {"n": 0, "frac_ood": 0.0, "psi": 0.0, "band": "no-data",
                "message": "No drift-scored inspections yet — process some images first."}

    arr = np.asarray(scores, dtype=float)
    counts, _ = np.histogram(arr, bins=bin_edges)
    actual = counts / counts.sum()
    psi = population_stability_index(expected, actual)
    frac_ood = float(np.mean(arr >= threshold))
    band = _band(psi)
    message = (
        f"Line drift: PSI {psi:.2f} — {band.upper()}; {int(np.sum(arr >= threshold))}/{len(scores)} parts OOD."
    )
    return {"n": len(scores), "frac_ood": round(frac_ood, 4), "psi": round(psi, 4), "band": band, "message": message}


def main() -> None:
    from memory import mes
    from memory import seed as seed_module

    seed_module.ensure_seeded()
    conn = mes.connect()
    try:
        report = population_report(conn)
    finally:
        conn.close()
    print(json.dumps(report, indent=2))
    print(f"[drift.report] {report['message'] if report['n'] else report.get('message', 'no data')}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `unset VIRTUAL_ENV; uv run python -m pytest tests/test_drift_report.py -v`
Expected: PASS (all four tests in the file).

- [ ] **Step 5: Commit**

```bash
git add drift/report.py tests/test_drift_report.py
git commit -m "feat(drift): add MES-backed windowed PSI population report"
```

---

## Task 12: `eval/drift_eval.py` — synthesize drift + calibrate threshold

**Files:**
- Create: `eval/drift_eval.py`
- Test: `tests/test_drift_scoring.py` (calibration helper unit test) + manual end-to-end run

- [ ] **Step 1: Write the failing test (pure calibration helper)**

Append to `tests/test_drift_scoring.py`:

```python
from eval.drift_eval import calibrate_threshold, psi_reference_bins


def test_calibrate_threshold_bounds_clean_false_alarm():
    # 100 clean scores 0..0.99; with a 5% alarm budget the threshold should sit near the 95th pct.
    clean = np.linspace(0.0, 0.99, 100)
    t = calibrate_threshold(clean, far_alarm_target=0.05)
    # At most ~5% of clean scores may exceed the threshold.
    assert np.mean(clean >= t) <= 0.05 + 1e-9


def test_psi_reference_bins_sum_to_one():
    clean = np.linspace(0.0, 1.0, 50)
    ref = psi_reference_bins(clean, n_bins=8)
    assert len(ref["bin_edges"]) == 9
    assert abs(sum(ref["expected_props"]) - 1.0) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `unset VIRTUAL_ENV; uv run python -m pytest tests/test_drift_scoring.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'eval.drift_eval'`.

- [ ] **Step 3: Implement the eval**

Create `eval/drift_eval.py`:

```python
"""Drift-monitor validation + threshold calibration (mirrors eval/perception_eval.py).

Methodology (honest, seeded, small-sample-aware):
  * Clean (in-distribution) set = the category's ``test/good`` images — DISJOINT from the reference
    (built on ``train/good``), so distances aren't artificially deflated.
  * Synthesize drift by perturbing copies of the clean images: brightness, contrast, gaussian blur,
    gaussian noise, JPEG compression (each at seeded severities).
  * Calibrate the OOD threshold on a seeded clean calibration split to bound the clean false-alarm
    rate at ``settings.drift_far_alarm_target``; report the alarm rate on the disjoint clean holdout
    with a Wilson 95% upper bound.
  * Report separability AUROC (clean vs drifted), per-perturbation detection rate, and the PSI
    reference bins consumed by drift/report.py.

Outputs:
  - artifacts/drift/<category>/drift_metrics.json
  - artifacts/drift/<category>/drift_separation.png

Run:  uv run python -m eval.drift_eval
"""
from __future__ import annotations

import io
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.metrics import roc_auc_score

from config import settings
from drift.reference import load_reference
from drift.scoring import knn_distance

CALIBRATION_FRACTION = 0.5


def _perturbations(rng: np.random.Generator) -> dict:
    """Name -> function(img: PIL.Image) -> PIL.Image. Deterministic given the seeded rng."""
    def brightness_down(im):
        return ImageEnhance.Brightness(im).enhance(0.55)

    def brightness_up(im):
        return ImageEnhance.Brightness(im).enhance(1.6)

    def contrast_down(im):
        return ImageEnhance.Contrast(im).enhance(0.5)

    def blur(im):
        return im.filter(ImageFilter.GaussianBlur(radius=2.5))

    def noise(im):
        arr = np.asarray(im, dtype=np.float32)
        arr = arr + rng.normal(0.0, 25.0, size=arr.shape)
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    def jpeg(im):
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=20)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    return {
        "brightness_down": brightness_down,
        "brightness_up": brightness_up,
        "contrast_down": contrast_down,
        "blur": blur,
        "noise": noise,
        "jpeg": jpeg,
    }


def calibrate_threshold(clean_scores: np.ndarray, far_alarm_target: float) -> float:
    """Smallest threshold whose clean false-alarm rate (fraction of clean scores >= t) <= target.

    Implemented as the (1 - target) quantile of the clean scores: at most ``target`` fraction of
    clean images exceed it.
    """
    return float(np.quantile(clean_scores, 1.0 - far_alarm_target))


def psi_reference_bins(clean_scores: np.ndarray, n_bins: int = 10) -> dict:
    """Histogram the clean (in-distribution) scores into n_bins; return edges + expected props.

    Edges span [min, max] of the clean scores with the outer edges pushed to ±inf so live scores
    beyond the observed clean range still fall into the end bins.
    """
    lo, hi = float(clean_scores.min()), float(clean_scores.max())
    if hi <= lo:
        hi = lo + 1e-6
    inner = np.linspace(lo, hi, n_bins + 1)
    edges = inner.copy()
    edges[0], edges[-1] = -np.inf, np.inf
    counts, _ = np.histogram(clean_scores, bins=edges)
    props = counts / counts.sum()
    # Store finite edges (replace ±inf with the observed bounds) for JSON + np.histogram reuse.
    finite_edges = inner.tolist()
    finite_edges[0], finite_edges[-1] = -1e9, 1e9
    return {"bin_edges": finite_edges, "expected_props": props.tolist()}


def _wilson_upper(k: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 1.0
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return float(min(1.0, center + half))


def _clean_image_paths() -> list[Path]:
    good = settings.data_root / "MVTecAD" / settings.category / "test" / "good"
    if not good.is_dir():
        raise FileNotFoundError(f"Clean set not found at {good}. Run perception.train first.")
    return sorted(good.glob("*.png"))


def _save_plot(clean: np.ndarray, drifted: np.ndarray, t: float, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(clean, bins=20, alpha=0.6, label="clean (in-distribution)", color="#2980b9", density=True)
    ax.hist(drifted, bins=20, alpha=0.6, label="drifted (synthetic)", color="#c0392b", density=True)
    ax.axvline(t, ls=":", color="black", label=f"OOD threshold = {t:.3f}")
    ax.set_xlabel("Drift score (mean kNN distance to training-good)")
    ax.set_ylabel("Density")
    ax.set_title(f"Drift-score separation — MVTec '{settings.category}'")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def evaluate() -> dict:
    rng = np.random.default_rng(settings.seed)
    reference = load_reference(settings.drift_reference_path)
    if reference.category != settings.category:
        raise ValueError(f"Reference is for '{reference.category}', not active '{settings.category}'.")

    from drift.extractor import EmbeddingExtractor

    extractor = EmbeddingExtractor()
    paths = _clean_image_paths()
    print(f"[drift.eval] Scoring {len(paths)} clean images + perturbations for '{settings.category}' (CPU)...")

    def score(img: Image.Image) -> float:
        return knn_distance(extractor.embed(img), reference.embeddings, settings.drift_k)

    clean_scores = np.array([score(Image.open(p).convert("RGB")) for p in paths], dtype=float)

    perts = _perturbations(rng)
    drifted_by_type: dict[str, np.ndarray] = {}
    for name, fn in perts.items():
        scores = []
        for p in paths:
            scores.append(score(fn(Image.open(p).convert("RGB"))))
        drifted_by_type[name] = np.array(scores, dtype=float)
    drifted_all = np.concatenate(list(drifted_by_type.values()))

    # Separability AUROC (threshold-free headline): clean=0, drifted=1.
    y = np.concatenate([np.zeros(len(clean_scores)), np.ones(len(drifted_all))])
    s = np.concatenate([clean_scores, drifted_all])
    auroc = float(roc_auc_score(y, s))

    # Calibrate on a seeded clean split; report alarm rate on the disjoint clean holdout.
    idx = np.arange(len(clean_scores))
    rng.shuffle(idx)
    n_cal = max(1, min(len(idx) - 1, int(round(len(idx) * CALIBRATION_FRACTION))))
    cal_idx, hold_idx = idx[:n_cal], idx[n_cal:]
    threshold = calibrate_threshold(clean_scores[cal_idx], settings.drift_far_alarm_target)

    hold_alarms = int(np.sum(clean_scores[hold_idx] >= threshold))
    n_hold = len(hold_idx)
    false_alarm_rate = hold_alarms / n_hold if n_hold else 0.0
    false_alarm_wilson = _wilson_upper(hold_alarms, n_hold)

    detection_rate = {name: float(np.mean(sc >= threshold)) for name, sc in drifted_by_type.items()}
    psi_ref = psi_reference_bins(clean_scores, n_bins=10)

    alarm_ok = bool(false_alarm_rate <= settings.drift_far_alarm_target)
    auroc_ok = bool(auroc >= 0.90)

    metrics = {
        "category": settings.category,
        "seed": settings.seed,
        "drift_k": settings.drift_k,
        "n_clean": int(len(clean_scores)),
        "n_drifted": int(len(drifted_all)),
        "separability_auroc": round(auroc, 4),
        "operating_threshold": round(threshold, 6),
        "far_alarm_target": settings.drift_far_alarm_target,
        "holdout": {
            "n_clean": n_hold,
            "false_alarm_rate": round(false_alarm_rate, 4),
            "false_alarm_wilson_upper95": round(false_alarm_wilson, 4),
        },
        "detection_rate_by_perturbation": {k: round(v, 4) for k, v in detection_rate.items()},
        "psi_reference": psi_ref,
        "auroc_ok": auroc_ok,
        "alarm_ok": alarm_ok,
        "methodology_note": (
            "Clean = test/good (disjoint from the train/good reference). Drift synthesized via "
            "brightness/contrast/blur/noise/jpeg perturbations. Threshold calibrated on a seeded clean "
            "split to bound the clean false-alarm rate; reported on the disjoint clean holdout with a "
            "Wilson upper bound. AUROC is threshold-free."
        ),
    }

    settings.drift_dir.mkdir(parents=True, exist_ok=True)
    settings.drift_metrics_path.write_text(json.dumps(metrics, indent=2))
    _save_plot(clean_scores, drifted_all, threshold, settings.drift_dir / "drift_separation.png")

    print("[drift.eval] Drift metrics:")
    print(json.dumps(metrics, indent=2))
    print(
        f"[drift.eval] separability AUROC {auroc:.4f} (>=0.90: {auroc_ok}); "
        f"clean holdout false-alarm {false_alarm_rate:.1%} "
        f"(Wilson-upper {false_alarm_wilson:.1%}; budget {settings.drift_far_alarm_target:.0%}: {alarm_ok})."
    )
    return metrics


if __name__ == "__main__":
    evaluate()
```

- [ ] **Step 4: Run the unit test to verify it passes**

Run: `unset VIRTUAL_ENV; uv run python -m pytest tests/test_drift_scoring.py -v`
Expected: PASS (calibration helper tests green).

- [ ] **Step 5: Commit**

```bash
git add eval/drift_eval.py tests/test_drift_scoring.py
git commit -m "feat(drift): add synthetic-drift validation + OOD threshold calibration"
```

---

## Task 13: `tests/test_drift_regression.py` — artifact-gated regression

**Files:**
- Create: `tests/test_drift_regression.py`
- Modify: `pyproject.toml` (add `drift` marker)

- [ ] **Step 1: Add the marker**

In `pyproject.toml`, under `[tool.pytest.ini_options].markers`, add:

```toml
    "drift: validates the drift-monitor calibration artifact (skipped if absent)",
```

- [ ] **Step 2: Write the test**

Create `tests/test_drift_regression.py`:

```python
"""Drift-monitor regression: validates drift_metrics.json against budgets.

Skips cleanly if the calibration artifact hasn't been produced yet (run
`uv run python -m drift.reference` then `uv run python -m eval.drift_eval`)."""
from __future__ import annotations

import json

import pytest

from config import settings

pytestmark = pytest.mark.drift

requires_artifact = pytest.mark.skipif(
    not settings.drift_metrics_path.exists(),
    reason="Drift artifact missing; run drift.reference + eval.drift_eval first.",
)


@requires_artifact
def test_separability_auroc_meets_budget():
    metrics = json.loads(settings.drift_metrics_path.read_text())
    assert metrics["category"] == settings.category
    assert metrics["separability_auroc"] >= 0.90, f"AUROC {metrics['separability_auroc']} below 0.90"


@requires_artifact
def test_clean_false_alarm_within_budget():
    metrics = json.loads(settings.drift_metrics_path.read_text())
    far = metrics["holdout"]["false_alarm_rate"]
    assert far <= settings.drift_far_alarm_target + 1e-9, (
        f"Clean false-alarm rate {far} exceeds budget {settings.drift_far_alarm_target}"
    )


@requires_artifact
def test_every_perturbation_detected_above_chance():
    metrics = json.loads(settings.drift_metrics_path.read_text())
    for name, rate in metrics["detection_rate_by_perturbation"].items():
        assert rate > 0.5, f"perturbation '{name}' detected at only {rate:.0%} (<= chance)"
```

- [ ] **Step 3: Run to verify it skips (artifact not built yet)**

Run: `unset VIRTUAL_ENV; uv run python -m pytest tests/test_drift_regression.py -v`
Expected: 3 SKIPPED (artifact absent). This is correct pre-build behavior, mirroring `test_perception_regression`.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml tests/test_drift_regression.py
git commit -m "test(drift): add artifact-gated drift regression suite + pytest marker"
```

---

## Task 14: Service — `/health` fields and `GET /drift`

**Files:**
- Modify: `service/app.py`
- Test: `tests/test_service.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_service.py`:

```python
@requires
def test_health_exposes_drift_fields(client):
    body = client.get("/health").json()
    assert "drift_enabled" in body
    assert "drift_reference_present" in body


@requires
def test_drift_report_endpoint(client):
    r = client.get("/drift")
    assert r.status_code == 200
    body = r.json()
    assert "band" in body and "n" in body
```

- [ ] **Step 2: Run test to verify it fails**

Run: `unset VIRTUAL_ENV; VQC_LLM_PROVIDER=stub uv run python -m pytest tests/test_service.py -v`
Expected: FAIL — `/health` lacks the new keys and `/drift` is 404 (or SKIPPED if model/MES absent — if skipped, build artifacts via the Task 16 commands first, then re-run).

- [ ] **Step 3: Update the service**

In `service/app.py`, extend `/health`:

```python
@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "category": settings.category,
        "llm_provider": settings.llm_provider,
        "confidence_threshold": settings.confidence_threshold,
        "drift_enabled": settings.drift_enabled,
        "drift_reference_present": settings.drift_reference_path.exists() and settings.drift_metrics_path.exists(),
    }
```

Add a `/drift` endpoint (after `/health`):

```python
@app.get("/drift")
def drift() -> dict:
    """Windowed population drift report (PSI + %OOD over recent drift-scored inspections)."""
    if not settings.drift_metrics_path.exists():
        raise HTTPException(status_code=503, detail="Drift monitor not calibrated; run drift.reference + eval.drift_eval.")
    from drift.report import population_report

    conn = mes.connect()
    try:
        return population_report(conn)
    finally:
        conn.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `unset VIRTUAL_ENV; VQC_LLM_PROVIDER=stub uv run python -m pytest tests/test_service.py -v`
Expected: PASS (skips only if the model/MES aren't present; build them via Task 16 then re-run).

- [ ] **Step 5: Commit**

```bash
git add service/app.py tests/test_service.py
git commit -m "feat(drift): expose drift status on /health and a /drift report endpoint"
```

---

## Task 15: UI — drift badge + line-drift panel

**Files:**
- Modify: `ui/streamlit_app.py`

- [ ] **Step 1: Add the per-inspection drift badge**

In `ui/streamlit_app.py`, inside the result block, after the `st.info(out.summary)` line, add:

```python
        if out.drift is not None:
            d = out.drift
            if d.is_ood:
                st.warning(
                    f"⚠️ Input drift: **OUT-OF-DISTRIBUTION** — {d.note} "
                    f"(drift score {d.drift_score:.3f} ≥ threshold {d.threshold:.3f}). "
                    "This image is outside the model's validated envelope, so the decision was escalated."
                )
            else:
                st.success(f"✅ Input in-distribution (drift score {d.drift_score:.3f} < threshold {d.threshold:.3f}).")
            st.caption(
                f"Brightness {d.brightness_delta:+.1f}σ · Contrast {d.contrast_delta:+.1f}σ · Sharpness {d.sharpness_delta:+.1f}σ "
                "(vs training baseline)"
            )
```

- [ ] **Step 2: Add the line-drift panel**

Still in `ui/streamlit_app.py`, add a helper near `part_context_caption` (top-level):

```python
def line_drift_report():
    from config import settings as _settings

    if not _settings.drift_metrics_path.exists():
        return None
    from drift.report import population_report

    conn = mes.connect()
    try:
        return population_report(conn)
    finally:
        conn.close()
```

And at the very end of the `with col_out:` block (after the investigation-trace expander, still inside `col_out`), add:

```python
    st.divider()
    st.subheader("Line drift monitor")
    report = line_drift_report()
    if report is None:
        st.caption("Drift monitor not calibrated — run `uv run python -m drift.reference` then `uv run python -m eval.drift_eval`.")
    elif report["n"] == 0:
        st.caption("No drift-scored inspections yet. Inspect a few uploaded images to populate the monitor.")
    else:
        band_icon = {"stable": "🟢", "moderate": "🟡", "significant": "🔴"}.get(report["band"], "•")
        st.metric("Population drift (PSI)", f"{report['psi']:.2f}", help="<0.1 stable · 0.1–0.25 moderate · >0.25 significant")
        st.write(f"{band_icon} **{report['band'].upper()}** — {report['frac_ood']:.0%} of the last {report['n']} parts flagged OOD.")
```

- [ ] **Step 3: Smoke-check the UI imports**

Run: `unset VIRTUAL_ENV; uv run python -c "import ast; ast.parse(open('ui/streamlit_app.py').read()); print('ok')"`
Expected: `ok` (syntax valid). Full visual check happens during the manual demo in Task 16.

- [ ] **Step 4: Commit**

```bash
git add ui/streamlit_app.py
git commit -m "feat(drift): show per-image drift badge and line-drift monitor in the UI"
```

---

## Task 16: Build artifacts, full-suite verification, docs, and PR

**Files:**
- Modify: `README.md`, `CLAUDE.md`, `PROJECT_STATUS.md` (gitignored — edit in place, do not commit)

- [ ] **Step 1: Build the drift reference + calibration for `bottle`**

Run:
```bash
unset VIRTUAL_ENV
uv run python -m drift.reference
uv run python -m eval.drift_eval
```
Expected: a reference saved under `artifacts/drift/bottle/reference.npz`, then `drift_metrics.json` + `drift_separation.png`, with a printed `separability AUROC ... (>=0.90: True)` and `false-alarm ... budget 5%: True`. If either budget reads `False`, STOP and report the metrics — do not loosen the asserts; investigate (e.g. perturbation severities, k) per the spec's success criteria.

- [ ] **Step 2: Build the drift artifacts for `hazelnut`**

Run:
```bash
unset VIRTUAL_ENV
VQC_CATEGORY=hazelnut uv run python -m drift.reference
VQC_CATEGORY=hazelnut uv run python -m eval.drift_eval
```
Expected: `artifacts/drift/hazelnut/...` produced with both budgets met.

- [ ] **Step 3: Run the full test suite (hermetic)**

Run: `unset VIRTUAL_ENV; VQC_LLM_PROVIDER=stub uv run python -m pytest`
Expected: all prior tests PASS plus the new drift tests; the drift regression now runs (not skipped) and passes. Report the final counts.

- [ ] **Step 4: Run the drift report end-to-end**

Run: `unset VIRTUAL_ENV; uv run python -m drift.report`
Expected: a JSON report. With only `qc` seed rows (no drift scores), `band` is `no-data` until real inspections run — that's expected; note it.

- [ ] **Step 5: Update README and CLAUDE.md**

In `README.md`, add a short "Drift monitoring" subsection documenting: what it does (per-image OOD escalation + MES-backed PSI monitor), the run commands (`drift.reference`, `eval.drift_eval`, `drift.report`), and the headline `separability_auroc` / false-alarm numbers from Steps 1–2.

In `CLAUDE.md`, update the "Monitor" line and "Limitations and Future Work" to reflect that input-distribution drift monitoring is now **implemented** (per-image kNN OOD gate + PSI population monitor), and add to "Dev notes": `drift/` artifacts live under `artifacts/drift/<category>/`; build with `drift.reference` + `eval.drift_eval`; OOD parts escalate.

- [ ] **Step 6: Update PROJECT_STATUS.md (gitignored — do NOT git add it)**

Add a "drift monitor — done" row to the milestone table and note the headline numbers + new run commands. This file is gitignored (`.gitignore` line 20); edit in place only.

- [ ] **Step 7: Commit docs**

```bash
git add README.md CLAUDE.md
git commit -m "docs(drift): document input-distribution drift monitoring"
```

- [ ] **Step 8: Push the branch and open a PR**

Run:
```bash
git push -u origin drift-monitor
gh pr create --title "Input-distribution drift monitor" --body "$(cat <<'EOF'
## Summary
- Per-image OOD gate: kNN distance to a per-category training-good reference embedding set; OOD parts escalate to a human, drift read annotated on every inspection.
- MES-backed PSI population monitor over the recent drift-scored stream.
- Validated like perception: synthetic-drift separability AUROC + clean false-alarm budget (Wilson CI), per-perturbation detection rates.

## Test
- Full suite green via `VQC_LLM_PROVIDER=stub uv run python -m pytest` (incl. new drift unit/integration/regression tests).
- Drift artifacts built and validated for `bottle` and `hazelnut`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```
Expected: PR created on `origin`. Report the URL.

---

## Self-Review (completed during planning)

**Spec coverage** — every spec section maps to a task: §3 architecture → Tasks 3–7, 10; §4 scoring/contract → Tasks 2–7; §5 loop integration/escalation/error-handling → Tasks 9–10; §6 population monitor + MES column → Tasks 8, 11; §7 validation → Task 12; §7 success criteria → Tasks 12–13, 16; §8 config → Task 1; §9 service+UI → Tasks 14–15; §10 tests → Tasks 3–14; §11 run commands → Task 16.

**Placeholder scan** — no TBD/TODO; every code step shows complete code; every command shows expected output.

**Type consistency** — `DriftResult` fields (`is_ood`, `drift_score`, `threshold`, `*_delta`, `note`) are identical across Tasks 2, 7, 9, 10, 15. `knn_distance(embedding, reference, k)` and `population_stability_index(expected, actual)` signatures match across Tasks 3, 7, 11, 12. `drift_metrics.json` keys (`operating_threshold`, `psi_reference.{bin_edges,expected_props}`, `separability_auroc`, `holdout.false_alarm_rate`, `detection_rate_by_perturbation`) are written in Task 12 and read identically in Tasks 11, 13, 14. `record_inspection(..., drift_score=...)` signature matches between Tasks 8 and 10. `get_drift_monitor()` defined in Task 9, used in Task 10.
