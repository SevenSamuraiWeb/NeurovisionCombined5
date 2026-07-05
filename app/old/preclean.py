"""Pre-clean stage: SCUNet (Swin-Conv-UNet, ECCV 2022) for blind real-world denoise.

Runs BEFORE super-resolution so noise/grain/JPEG-blocky artifacts are removed
rather than sharpened by Real-ESRGAN. SCUNet's `color_real_psnr` weights are
trained on a degradation pipeline that combines Gaussian noise, ISO noise,
JPEG compression, and resampling — so a single pass covers most legacy-scan
artifacts.

A separate dedicated JPEG-artifact net (FBCNN) is left as a follow-up. SCUNet
alone removes most JPEG blocking we see on scanned photos.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

_VENDORED_ROOT = Path(__file__).resolve().parent / "vendored"
if str(_VENDORED_ROOT) not in sys.path:
    sys.path.insert(0, str(_VENDORED_ROOT))

from scunet.network_scunet import SCUNet  # noqa: E402


_SCUNET_WEIGHTS_URL = (
    "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth"
)
_WEIGHTS_DIR = Path(
    os.environ.get(
        "NV_PRECLEAN_WEIGHTS_DIR",
        str(Path(__file__).resolve().parent.parent.parent / "models"),
    )
)


def _download_if_missing(url: str, dst: Path) -> Path:
    if dst.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    from urllib.request import urlretrieve

    print(f"[preclean] downloading {url}", flush=True)
    urlretrieve(url, dst)
    return dst


class PreCleanAdapter:
    """SCUNet blind-degradation cleanup, tile-based for large inputs."""

    def __init__(
        self,
        device: Optional[str] = None,
        tile: int = 512,
        tile_overlap: int = 32,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tile = tile
        self.tile_overlap = tile_overlap

        weight_path = _download_if_missing(
            _SCUNET_WEIGHTS_URL, _WEIGHTS_DIR / "scunet_color_real_psnr.pth"
        )
        self.net = SCUNet(in_nc=3, config=[4, 4, 4, 4, 4, 4, 4], dim=64)
        state = torch.load(weight_path, map_location=self.device, weights_only=False)
        self.net.load_state_dict(state, strict=True)
        self.net.to(self.device).eval()

    @torch.no_grad()
    def _infer_tile(self, tile_rgb: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(tile_rgb.astype(np.float32) / 255.0)
        t = t.permute(2, 0, 1).unsqueeze(0).to(self.device)
        out = self.net(t)
        out = out.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
        return (out * 255.0 + 0.5).clip(0, 255).astype(np.uint8)

    def _denoise_tiled(self, rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        tile = self.tile
        if h <= tile and w <= tile:
            return self._infer_tile(rgb)

        ov = self.tile_overlap
        stride = tile - ov
        out = np.zeros_like(rgb, dtype=np.float32)
        weight = np.zeros((h, w, 1), dtype=np.float32)

        y_starts = list(range(0, max(h - tile, 0) + 1, stride)) or [0]
        x_starts = list(range(0, max(w - tile, 0) + 1, stride)) or [0]
        if y_starts[-1] + tile < h:
            y_starts.append(h - tile)
        if x_starts[-1] + tile < w:
            x_starts.append(w - tile)

        # Cosine fade at tile edges to suppress seams.
        feather = np.ones((tile, tile), dtype=np.float32)
        ramp = np.linspace(0.0, 1.0, ov, dtype=np.float32)
        feather[:ov, :] *= ramp[:, None]
        feather[-ov:, :] *= ramp[::-1][:, None]
        feather[:, :ov] *= ramp[None, :]
        feather[:, -ov:] *= ramp[::-1][None, :]
        feather = feather[..., None]

        for y in y_starts:
            for x in x_starts:
                tile_in = rgb[y : y + tile, x : x + tile]
                # Pad if the last tile is smaller than `tile`
                ph, pw = tile_in.shape[:2]
                if ph < tile or pw < tile:
                    padded = np.zeros((tile, tile, 3), dtype=rgb.dtype)
                    padded[:ph, :pw] = tile_in
                    tile_in = padded
                tile_out = self._infer_tile(tile_in)
                f = feather[:ph, :pw]
                out[y : y + ph, x : x + pw] += tile_out[:ph, :pw].astype(np.float32) * f
                weight[y : y + ph, x : x + pw] += f

        out = out / np.maximum(weight, 1e-8)
        return out.clip(0, 255).astype(np.uint8)

    def clean(self, image: Image.Image) -> Image.Image:
        arr = np.array(image.convert("RGB"))
        cleaned = self._denoise_tiled(arr)
        return Image.fromarray(cleaned)


def needs_preclean(
    predicted_types: list[str], severity: float, threshold: float = 0.4
) -> bool:
    """Cheap routing decision: skip pre-clean on near-clean inputs."""
    if "noise" in predicted_types:
        return True
    return severity > threshold
