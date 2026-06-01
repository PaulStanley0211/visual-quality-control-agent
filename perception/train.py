"""Train PatchCore on a single MVTec AD category (CPU) and export a Torch model.

Run:  uv run python -m perception.train

PatchCore is a feature-memory-bank method: there is no gradient training, so a
single pass over the normal ("good") training images on CPU is sufficient. The
configured category (default ``bottle``) is auto-downloaded on first run.

The exported Torch model is what the detector and the FastAPI service load at
inference time (via ``TorchInferencer``) — decoupled from the training Engine.
"""
from __future__ import annotations

from pathlib import Path

from anomalib.data import MVTecAD
from anomalib.deploy import ExportType
from anomalib.engine import Engine
from anomalib.models import Patchcore
from lightning.pytorch import seed_everything

from config import settings
from perception.prepare_data import prepare


def train() -> Path:
    """Fit PatchCore on the configured category and export a Torch model.

    Returns the path to the exported ``model.pt``.
    """
    seed_everything(settings.seed, workers=True)
    settings.perception_dir.mkdir(parents=True, exist_ok=True)

    # Ensure the category is present locally (works around anomalib's dead mirror).
    prepare()

    datamodule = MVTecAD(
        root=settings.data_root / "MVTecAD",
        category=settings.category,
        train_batch_size=8,
        eval_batch_size=8,
        num_workers=0,  # Windows + CPU: avoid dataloader worker spawn overhead/issues
        seed=settings.seed,  # reproducible train/val/test split
    )

    model = Patchcore(
        backbone=settings.backbone,
        layers=["layer2", "layer3"],
        pre_trained=True,
        coreset_sampling_ratio=settings.coreset_sampling_ratio,
    )

    engine = Engine(
        accelerator=settings.accelerator,
        devices=settings.devices,
        max_epochs=1,  # PatchCore builds a coreset memory bank in a single pass; always 1
        default_root_dir=str(settings.perception_dir),
    )

    print(f"[train] Fitting PatchCore ({settings.backbone}) on category '{settings.category}' (CPU)...")
    engine.fit(model=model, datamodule=datamodule)

    export_path = engine.export(
        model=model,
        export_type=ExportType.TORCH,
        export_root=str(settings.perception_dir),
    )
    print(f"[train] Exported Torch model to: {export_path}")
    return Path(export_path)


if __name__ == "__main__":
    train()
