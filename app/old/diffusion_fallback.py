"""Diffusion-based super-resolution fallback for severely degraded inputs.

When `compute_severity()` exceeds `_SEVERITY_THRESHOLD` and CUDA + dependencies
are available, the pipeline routes background SR through Stable Diffusion's
4x upscaler (Stability AI's `stable-diffusion-x4-upscaler`) instead of
Real-ESRGAN. Diffusion-based SR is much slower (~10-30s/image on a 24GB GPU)
but handles severe degradation (heavy fade, large missing regions filled by
LaMa, generally unrecoverable detail) where GAN-based SR plateaus.

This is gated behind three independent checks:
  1. Environment: `NV_ENABLE_DIFFUSION=1`
  2. Runtime: torch.cuda.is_available()
  3. Per-image: severity > 0.65 (see severity.py)

If any check fails, the caller (pipeline.py) falls back to Real-ESRGAN.
Dependencies (`diffusers`, `accelerate`, optionally `xformers`) are imported
lazily so the default install has zero diffusion footprint.
"""
from __future__ import annotations

import os
from typing import Optional

import torch
from PIL import Image


_SEVERITY_THRESHOLD = 0.65
_MODEL_ID = os.environ.get("NV_DIFFUSION_MODEL", "stabilityai/stable-diffusion-x4-upscaler")


def diffusion_enabled() -> bool:
    """Cheap precheck before instantiating anything."""
    if os.environ.get("NV_ENABLE_DIFFUSION", "0") != "1":
        return False
    if not torch.cuda.is_available():
        return False
    return True


def should_use_diffusion(severity: float) -> bool:
    if not diffusion_enabled():
        return False
    return severity >= _SEVERITY_THRESHOLD


class DiffusionSRAdapter:
    """Stable Diffusion x4 upscaler wrapper. Instantiation is the heavy step;
    `enhance()` is per-image. Constructor raises if deps missing or no CUDA."""

    def __init__(
        self,
        model_id: str = _MODEL_ID,
        device: Optional[str] = None,
        use_xformers: bool = True,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("DiffusionSRAdapter requires CUDA.")
        try:
            from diffusers import StableDiffusionUpscalePipeline
        except ImportError as exc:
            raise RuntimeError(
                "DiffusionSRAdapter requires `pip install diffusers accelerate`."
            ) from exc

        self.device = torch.device(device or "cuda")
        self.pipeline = StableDiffusionUpscalePipeline.from_pretrained(
            model_id, torch_dtype=torch.float16
        ).to(self.device)
        self.pipeline.set_progress_bar_config(disable=True)
        # Memory savings (essential on <16GB GPUs).
        self.pipeline.enable_attention_slicing()
        self.pipeline.enable_vae_tiling()
        if use_xformers:
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
            except Exception:
                pass

    def enhance(
        self,
        image: Image.Image,
        prompt: str = "a high quality photograph, sharp details, natural textures",
        num_inference_steps: int = 30,
        guidance_scale: float = 5.0,
        noise_level: int = 20,
    ) -> Image.Image:
        """Run diffusion x4 upscale. Caps input long-edge at 512 to keep VRAM
        bounded; output is x4 of the capped input, then resized to match the
        original aspect ratio of a x4 of the true input."""
        original_size = image.size
        max_in = 512
        long_edge = max(original_size)
        if long_edge > max_in:
            scale = max_in / long_edge
            work = image.resize(
                (round(original_size[0] * scale), round(original_size[1] * scale)),
                Image.Resampling.LANCZOS,
            )
        else:
            work = image

        with torch.no_grad():
            out = self.pipeline(
                prompt=prompt,
                image=work,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                noise_level=noise_level,
            ).images[0]

        # Match the scale Real-ESRGAN would have produced (4x original).
        target = (original_size[0] * 4, original_size[1] * 4)
        if out.size != target:
            out = out.resize(target, Image.Resampling.LANCZOS)
        return out


def make_adapter_if_enabled() -> Optional[DiffusionSRAdapter]:
    """Return a constructed adapter if the env+CUDA gate is open; otherwise None.

    Catches construction errors (missing deps, OOM at load) and degrades to None
    so callers can fall back to Real-ESRGAN without a crash.
    """
    if not diffusion_enabled():
        return None
    try:
        return DiffusionSRAdapter()
    except Exception as exc:
        print(f"[diffusion] adapter init failed, falling back: {exc}", flush=True)
        return None
