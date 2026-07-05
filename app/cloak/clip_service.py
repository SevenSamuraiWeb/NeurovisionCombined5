"""
app/cloak/clip_service.py
=========================
Adversarial whole-image cloaking via CLIP embedding attacks.

Surrogate model  : openai/clip ViT-B/32 (reused from RAG/embedder.py)
Attack methods   : FGSM  |  PGD  |  MI-FGSM
Loss             : Untargeted: cosine_similarity(emb_adv, emb_clean)  → minimised
                   Targeted:   cosine_similarity(emb_adv, emb_target) → maximised
Metrics returned : SSIM, PSNR (dB), cosine similarity, semantic shift %

Normalization note
------------------
CLIP uses specific ImageNet-style normalization constants.
We keep perturbations in [0, 1] pixel space and apply CLIP normalization
only inside the forward pass to maintain a consistent gradient chain.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import clip

logger = logging.getLogger(__name__)

# Reusing the existing CLIP model from RAG to avoid double-loading
from RAG.embedder import model as _clip_model, preprocess as _clip_preprocess, device as _device

# CLIP's internal normalization constants
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=_device).view(1, 3, 1, 1)
CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=_device).view(1, 3, 1, 1)


def get_clip_status() -> dict:
    return {
        "loaded": _clip_model is not None,
        "device": str(_device),
        "surrogate": "CLIP-ViT-B/32",
    }


# ── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class ClipShieldResult:
    cloaked_image: Image.Image
    ssim: float
    psnr_db: float
    cosine_sim_before: float
    cosine_sim_after: float
    semantic_shift_pct: float
    noise_heatmap: Optional[Image.Image] = None
    method: str = "mifgsm"
    epsilon_used: float = 8 / 255
    steps: int = 20
    target_prompt: Optional[str] = None
    warning: Optional[str] = None


# ── Normalisation & Embedding helper ─────────────────────────────────────────

def _clip_embed(t: torch.Tensor) -> torch.Tensor:
    """
    t : float32 tensor in [0, 1], shape (1, 3, 224, 224)
    Returns L2-normalized CLIP image embedding.
    """
    t_norm = (t - CLIP_MEAN) / CLIP_STD
    emb = _clip_model.encode_image(t_norm)
    return F.normalize(emb, dim=-1)


def _clip_embed_text(text: str) -> torch.Tensor:
    """Returns L2-normalized CLIP text embedding."""
    tokens = clip.tokenize([text]).to(_device)
    emb = _clip_model.encode_text(tokens)
    return F.normalize(emb, dim=-1)


# ── Attack implementations ───────────────────────────────────────────────────

def _fgsm_shield(
    x: torch.Tensor,
    target_emb: torch.Tensor,
    epsilon: float,
    targeted: bool = False,
) -> torch.Tensor:
    """
    Single-step FGSM.
    If targeted: maximise similarity to target_emb (descend -similarity).
    If untargeted: minimise similarity to clean_emb (descend similarity).
    """
    x_in = x.clone().detach().requires_grad_(True)
    sim = F.cosine_similarity(_clip_embed(x_in), target_emb).mean()
    loss = -sim if targeted else sim
    grad = torch.autograd.grad(loss, x_in)[0]
    
    with torch.no_grad():
        x_adv = x_in - epsilon * grad.sign()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv.detach()


def _pgd_shield(
    x: torch.Tensor,
    target_emb: torch.Tensor,
    epsilon: float,
    alpha: float,
    steps: int,
    targeted: bool = False,
) -> torch.Tensor:
    x_adv = x.clone().detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        sim = F.cosine_similarity(_clip_embed(x_adv), target_emb).mean()
        loss = -sim if targeted else sim
        grad = torch.autograd.grad(loss, x_adv)[0]
        
        with torch.no_grad():
            x_adv = x_adv.detach() - alpha * grad.sign()
            x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv.detach()


def _mifgsm_shield(
    x: torch.Tensor,
    target_emb: torch.Tensor,
    epsilon: float,
    alpha: float,
    steps: int,
    decay: float = 1.0,
    targeted: bool = False,
) -> torch.Tensor:
    x_adv = x.clone().detach()
    g = torch.zeros_like(x)
    for _ in range(steps):
        x_adv.requires_grad_(True)
        sim = F.cosine_similarity(_clip_embed(x_adv), target_emb).mean()
        loss = -sim if targeted else sim
        grad = torch.autograd.grad(loss, x_adv)[0]
        
        with torch.no_grad():
            grad_norm = grad.abs().sum() + 1e-12
            g = decay * g + grad / grad_norm
            x_adv = x_adv.detach() - alpha * g.sign()
            x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv.detach()


# ── Metrics helpers ──────────────────────────────────────────────────────────

def _compute_ssim_psnr(orig: np.ndarray, cloaked: np.ndarray) -> tuple[float, float]:
    from skimage.metrics import (
        peak_signal_noise_ratio as _psnr,
        structural_similarity as _ssim,
    )
    if np.array_equal(orig, cloaked):
        return 1.0, 100.0

    ssim_val = float(_ssim(orig, cloaked, channel_axis=2, data_range=255))
    import math
    psnr_val = float(_psnr(orig, cloaked, data_range=255))
    if math.isinf(psnr_val):
        psnr_val = 100.0
    return ssim_val, psnr_val


# ── Main entry point ─────────────────────────────────────────────────────────

def shield_image(
    pil_image: Image.Image,
    epsilon: float = 8 / 255,
    method: str = "mifgsm",
    steps: int = 20,
    alpha: float = 2 / 255,
    decay: float = 1.0,
    target_prompt: Optional[str] = None,
) -> ClipShieldResult:
    """
    CLIP semantic shield pipeline.
    """
    method = method.lower().strip()
    if method not in ("fgsm", "pgd", "mifgsm"):
        raise ValueError(f"Unknown method '{method}'. Choose fgsm | pgd | mifgsm.")

    orig_size = pil_image.size  # (W, H)
    img_rgb = pil_image.convert("RGB")
    
    # 1. Resize to 224x224 for CLIP, convert to float [0, 1] tensor
    img_224 = img_rgb.resize((224, 224), Image.LANCZOS)
    orig_np_224 = np.array(img_224)
    x_orig = torch.tensor(orig_np_224, dtype=torch.float32, device=_device).permute(2, 0, 1).unsqueeze(0) / 255.0

    # 2. Get clean image embedding and target embedding
    _DEFAULT_TARGET = "a photo of something abstract or unrecognizable"
    with torch.no_grad():
        clean_emb = _clip_embed(x_orig).detach()
        if target_prompt:
            target_emb = _clip_embed_text(target_prompt).detach()
        else:
            # Always use a targeted attack so the embedding is pulled away from
            # the correct semantic class rather than just away from the clean image
            # embedding (which can leave it near the same text-label cluster).
            target_emb = _clip_embed_text(_DEFAULT_TARGET).detach()

    # 3. Attack — always targeted so the image embedding is actively steered
    # toward an unrecognizable region, not merely drifted away from clean.
    targeted = True
    if method == "fgsm":
        x_adv = _fgsm_shield(x_orig, target_emb, epsilon, targeted=targeted)
    elif method == "pgd":
        x_adv = _pgd_shield(x_orig, target_emb, epsilon, alpha, steps, targeted=targeted)
    else:
        x_adv = _mifgsm_shield(x_orig, target_emb, epsilon, alpha, steps, decay, targeted=targeted)

    # 4. Metrics at 224x224
    with torch.no_grad():
        adv_emb = _clip_embed(x_adv).detach()
        cos_sim_after = F.cosine_similarity(adv_emb, clean_emb).item()

    adv_np_224 = (x_adv.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)
    ssim_val, psnr_val = _compute_ssim_psnr(orig_np_224, adv_np_224)
    semantic_shift_pct = round((1.0 - cos_sim_after) * 100, 1)

    # 5. Upscale Delta and apply to original image
    # We upscale the difference instead of the image to preserve perturbation details
    delta_224 = adv_np_224.astype(np.float32) - orig_np_224.astype(np.float32)
    delta_shifted_uint8 = np.clip(delta_224 + 128.0, 0, 255).astype(np.uint8)
    delta_pil = Image.fromarray(delta_shifted_uint8)
    if delta_pil.mode != "RGB":
        delta_pil = delta_pil.convert("RGB")
    delta_upscaled_np = np.array(delta_pil.resize(orig_size, Image.BICUBIC)).astype(np.float32) - 128.0
    
    orig_np = np.array(img_rgb).astype(np.float32)
    cloaked_np = np.clip(orig_np + delta_upscaled_np, 0, 255).astype(np.uint8)
    cloaked_pil = Image.fromarray(cloaked_np)

    # 6. Noise Heatmap
    noise_amp = np.clip(np.abs(delta_224) * 10, 0, 255).astype(np.uint8)
    noise_heatmap = Image.fromarray(noise_amp)

    return ClipShieldResult(
        cloaked_image=cloaked_pil,
        ssim=round(ssim_val, 5),
        psnr_db=round(psnr_val, 3),
        cosine_sim_before=1.0,
        cosine_sim_after=round(cos_sim_after, 5),
        semantic_shift_pct=semantic_shift_pct,
        noise_heatmap=noise_heatmap,
        method=method,
        epsilon_used=epsilon,
        steps=steps,
        target_prompt=target_prompt,
        warning=None,
    )
