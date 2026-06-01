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
