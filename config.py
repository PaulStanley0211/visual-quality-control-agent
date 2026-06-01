"""Central configuration for the Visual Quality Control Agent.

Single source of truth for paths, the error-budget targets, routing thresholds,
and the (swappable) LLM provider. Every value is overridable via ``VQC_*``
environment variables or a local ``.env`` file, so the confidence threshold and
provider can be tuned at deploy time without code changes.
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VQC_", env_file=".env", extra="ignore")

    # --- paths ---
    data_root: Path = PROJECT_ROOT / "datasets"
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"
    mes_db_path: Path = PROJECT_ROOT / "memory" / "mes.db"

    # --- perception ---
    category: str = "bottle"
    image_size: int = 256  # matches anomalib's PatchCore pre-processor default (train + inference resize)
    backbone: str = "resnet18"  # lighter than the wide_resnet50_2 default; keeps CPU fit/inference fast

    # --- compute (threaded into the training Engine; override for GPU hosts) ---
    accelerator: str = "cpu"  # lightning accelerator: "cpu" | "gpu" | "auto"
    devices: int = 1

    # --- error budget ---
    far_target: float = 0.02  # max false-accept rate (defect passed as good) at the operating threshold
    agent_accuracy_target: float = 0.95  # disposition + random/systematic accuracy target

    # --- routing / escalation ---
    confidence_threshold: float = 0.60  # below this, low-confidence cases escalate to a human

    # --- investigation: systematic vs random ---
    systematic_defect_rate: float = 0.30  # machine/batch recent defect-rate above this => systematic
    history_window: int = 20  # how many recent parts define "recent" for the rate

    # --- reasoning LLM (offline-first, swappable) ---
    llm_provider: str = "stub"  # "stub" | "anthropic" | "ollama"
    anthropic_model: str = "claude-opus-4-8"
    ollama_model: str = "llama3.1"

    # --- reproducibility ---
    seed: int = 1337

    @property
    def perception_dir(self) -> Path:
        """Where the trained PatchCore checkpoint, exported model, and metrics live."""
        return self.artifacts_dir / "perception"

    @property
    def heatmaps_dir(self) -> Path:
        return self.artifacts_dir / "heatmaps"

    @property
    def metrics_path(self) -> Path:
        """Perception metrics + calibrated operating threshold, written by eval, read by the detector."""
        return self.perception_dir / "metrics.json"


settings = Settings()
