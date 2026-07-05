from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


DAMAGE_TYPES = [
    "noise",
    "blur",
    "fade",
    "sepia",
    "scratches",
    "missing_patch",
    "stains",
]

LOCALIZABLE_DAMAGE_TYPES = [
    "scratches",
    "missing_patch",
    "stains",
]

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_ROOT = Path(os.environ.get("NV_DATA_ROOT", str(_PROJECT_ROOT / "data" / "old")))
DEFAULT_GENERATED_ROOT = Path(
    os.environ.get("NV_GENERATED_ROOT", str(DEFAULT_DATA_ROOT / "generated_labeled"))
)
DEFAULT_MODEL_PATH = Path(
    os.environ.get("NV_MODEL_PATH", str(_PROJECT_ROOT / "models" / "old_photo_damage.pth"))
)
DEFAULT_RUNS_DIR = Path(
    os.environ.get("NV_RUNS_DIR", str(DEFAULT_DATA_ROOT / "pipeline_runs"))
)

RESTORATION_STEP_BY_DAMAGE = {
    "noise": "real_esrgan",
    "fade": "real_esrgan",
    "sepia": "real_esrgan",
    "scratches": "lama",
    "missing_patch": "lama",
    "stains": "lama",
}


def is_grayscale(image: Image.Image, chroma_thresh: float = 4.0) -> bool:
    """Detect B&W / sepia inputs via mean LAB chromaticity.

    `chroma_thresh` is in LAB a/b units (centered at 128); ~4 catches true B&W
    and most sepia/duotone scans while leaving lightly desaturated color
    photos alone.
    """
    arr = np.array(image.convert("RGB"))
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    ab = lab[..., 1:] - 128.0
    chroma = np.linalg.norm(ab, axis=2)
    return float(chroma.mean()) < chroma_thresh


def adaptive_tile_size() -> int:
    """Pick a Real-ESRGAN tile size from free CUDA memory. 0 = no tiling."""
    if not torch.cuda.is_available():
        return 400  # CPU: keep memory bounded
    try:
        free, _total = torch.cuda.mem_get_info()
    except RuntimeError:
        return 400
    gb = free / (1024 ** 3)
    if gb >= 12:
        return 0
    if gb >= 6:
        return 512
    if gb >= 4:
        return 384
    if gb >= 2:
        return 256
    return 128