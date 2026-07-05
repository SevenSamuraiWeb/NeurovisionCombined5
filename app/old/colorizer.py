"""B&W / sepia colorization using DDColor (CVPR 2023).

Vendored arch (`app/old/vendored/ddcolor`) + weights downloaded directly from
HuggingFace. ModelScope was tried first but its Windows download path is
fragile (os.rename race vs Defender on demo files). Direct HF download is
single-file and reliable.

Routing:
  - Auto-detect grayscale-ish inputs via `is_grayscale()` (mean LAB chroma).
  - Run colorization AFTER restoration so we colorize the sharpest image we
    output, not a noisy/blocky scan.

DeOldify ensemble dropped — official pin (fastai 1.0.61) does not install on
Python 3.12. DDColor alone is SOTA on FID/CF and good for portraits.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve

import cv2
import numpy as np
import torch
from PIL import Image

from app.old.common import is_grayscale
from app.old.vendored.ddcolor import DDColor


_DDCOLOR_WEIGHTS_URL = os.environ.get(
    "NV_DDCOLOR_WEIGHTS_URL",
    "https://huggingface.co/piddnad/DDColor-models/resolve/main/ddcolor_modelscope.pth",
)
_WEIGHTS_DIR = Path(
    os.environ.get(
        "NV_DDCOLOR_WEIGHTS_DIR",
        str(Path(__file__).resolve().parent.parent.parent / "models"),
    )
)


def _download_if_missing(url: str, dst: Path) -> Path:
    if dst.exists() and dst.stat().st_size > 1024 * 1024:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"[colorize] downloading {url}", flush=True)
    urlretrieve(url, dst)
    return dst


class ColorizationAdapter:
    """DDColor (ConvNeXt-L) wrapper. Lazy-loads weights on first use."""

    def __init__(
        self,
        device: Optional[str] = None,
        input_size: int = 512,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.input_size = input_size
        self._net: Optional[DDColor] = None

    def _ensure_loaded(self) -> DDColor:
        if self._net is not None:
            return self._net
        weight_path = _download_if_missing(
            _DDCOLOR_WEIGHTS_URL, _WEIGHTS_DIR / "ddcolor_modelscope.pth"
        )
        net = DDColor(
            encoder_name="convnext-l",
            decoder_name="MultiScaleColorDecoder",
            input_size=(self.input_size, self.input_size),
            num_output_channels=2,
            last_norm="Spectral",
            do_normalize=False,
            num_queries=100,
            num_scales=3,
            dec_layers=9,
        )
        state = torch.load(weight_path, map_location=self.device, weights_only=False)
        # Checkpoints from KAIR/modelscope use a `params` key
        if isinstance(state, dict) and "params" in state:
            state = state["params"]
        net.load_state_dict(state, strict=False)
        net.to(self.device).eval()
        self._net = net
        return net

    @torch.no_grad()
    def colorize(self, image: Image.Image) -> Image.Image:
        """Run DDColor using the official inference recipe.

        Critical: DDColor was trained against `cv2.cvtColor(..., COLOR_RGB2LAB)`
        on **float** RGB in [0, 1]. In OpenCV that yields L in [0, 100] and
        ab in [-128, 127] (centered at 0). Using uint8 LAB (L [0,255], ab
        [0,255] centered at 128) and rescaling with ``* 128 + 128`` is wrong
        and produces the magenta/blue artifact pattern.
        """
        net = self._ensure_loaded()
        rgb01 = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
        h0, w0 = rgb01.shape[:2]

        # Original L at full resolution — preserved verbatim in the output.
        orig_l = cv2.cvtColor(rgb01, cv2.COLOR_RGB2LAB)[:, :, :1]  # (H, W, 1), [0, 100]

        # Build a "gray RGB" at input_size for the network.
        resized = cv2.resize(rgb01, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        resized_l = cv2.cvtColor(resized, cv2.COLOR_RGB2LAB)[:, :, :1]
        gray_lab = np.concatenate(
            [resized_l, np.zeros_like(resized_l), np.zeros_like(resized_l)], axis=-1
        ).astype(np.float32)
        gray_rgb = cv2.cvtColor(gray_lab, cv2.COLOR_LAB2RGB)  # float, in [0, 1]

        tensor = torch.from_numpy(
            np.ascontiguousarray(gray_rgb.transpose(2, 0, 1))
        ).unsqueeze(0).to(self.device)

        # Forward — output is ab in LAB float range (centered at 0).
        output_ab = net(tensor).squeeze(0).float().cpu().numpy()  # (2, H, W)
        output_ab = output_ab.transpose(1, 2, 0)  # (H, W, 2)

        # Resize predicted ab back to the original resolution.
        ab_full = cv2.resize(output_ab, (w0, h0), interpolation=cv2.INTER_LINEAR)

        # Compose float-LAB and convert back to RGB.
        output_lab = np.concatenate([orig_l, ab_full], axis=-1).astype(np.float32)
        output_rgb = cv2.cvtColor(output_lab, cv2.COLOR_LAB2RGB)
        output_rgb = (np.clip(output_rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        return Image.fromarray(output_rgb)


def needs_colorize(image: Image.Image, chroma_thresh: float = 4.0) -> bool:
    return is_grayscale(image, chroma_thresh=chroma_thresh)
