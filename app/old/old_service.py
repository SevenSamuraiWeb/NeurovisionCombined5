from __future__ import annotations

import torch
from PIL import Image

from app.old.pipeline import OldPhotoRestorationPipeline

# Lazily instantiate the pipeline on first use. Constructing it eagerly at
# import time downloads model weights (CodeFormer via FaceRestorationAdapter),
# which crashes startup when offline or weights are absent.
_pipeline: OldPhotoRestorationPipeline | None = None


def _get_pipeline() -> OldPhotoRestorationPipeline:
    global _pipeline
    if _pipeline is None:
        use_gpu = torch.cuda.is_available()
        _pipeline = OldPhotoRestorationPipeline(device="cuda" if use_gpu else "cpu")
    return _pipeline


def loaded() -> bool:
    return _pipeline is not None


def de_old(image: Image.Image, report=None) -> Image.Image:
    """Old-photo restoration. report(substage_name, pil_image) receives
    intermediates ("input"/"final" are skipped — duplicates of what the
    caller already has)."""
    def on_progress(stage_name, img):
        if report and stage_name not in {"input", "final"}:
            report(stage_name, img)

    artifacts = _get_pipeline().run(image, on_progress=on_progress)
    return artifacts.memory_stages["final"]
