"""
app/cloak/cloak_service.py
==========================
Adversarial face cloaking via white-box embedding attacks.

Surrogate model  : facenet-pytorch  InceptionResnetV1 (VGGFace2 weights)
Face detector    : facenet-pytorch  MTCNN  (160×160 standardised crops)
Attack methods   : FGSM  |  PGD  |  MI-FGSM
Loss             : cosine_similarity(emb_adv, emb_clean)  → minimised
                   i.e. push the adversarial embedding AWAY from the
                   clean embedding (untargeted "dodging" attack).
Metrics returned : per-face  cosine_similarity, SSIM, PSNR (dB)

Normalization note
------------------
InceptionResnetV1 expects input in the range approximately [-1, 1]
produced by fixed_image_standardization:
    out = (x_0_255 - 127.5) / 128.0
We keep perturbations in [0, 1] pixel space and call
    fixed_image_standardization(t * 255.0)
inside embed() so the gradient chain is consistent.  Epsilon is therefore
specified in [0, 1] space (default 8/255 ≈ 0.0314).
"""

from __future__ import annotations
import io
import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)

# ── Lazy singletons ──────────────────────────────────────────────────────────
_mtcnn: Optional[object] = None
_resnet: Optional[object] = None
_device: str = "cpu"
_models_loaded: bool = False


def load_models() -> None:
    """
    Load MTCNN and InceptionResnetV1 once at blueprint registration time.
    Weights are downloaded automatically into the PyTorch cache on first call
    and reused from cache on subsequent calls.
    """
    global _mtcnn, _resnet, _device, _models_loaded

    if _models_loaded:
        return

    try:
        from facenet_pytorch import MTCNN, InceptionResnetV1  # noqa: PLC0415

        _device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("[cloak] Loading MTCNN on device=%s …", _device)
        _mtcnn = MTCNN(
            image_size=160,
            margin=14,
            keep_all=True,          # detect ALL faces
            device=_device,
        )
        logger.info("[cloak] Loading InceptionResnetV1 (vggface2) on device=%s …", _device)
        _resnet = InceptionResnetV1(pretrained="vggface2").eval().to(_device)
        _models_loaded = True
        logger.info("[cloak] Models ready.")
    except Exception as exc:
        logger.exception("[cloak] Model load failed: %s", exc)
        raise


def get_model_status() -> dict:
    return {
        "loaded": _models_loaded,
        "device": _device,
        "surrogate": "InceptionResnetV1-vggface2",
        "detector": "MTCNN",
    }


def get_mtcnn():
    """Return the loaded MTCNN singleton for reuse (e.g. auto-routing)."""
    return _mtcnn


# ── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class FaceMetrics:
    face_idx: int
    cosine_similarity: float    # lower = better protection
    ssim: float                 # closer to 1.0 = visually imperceptible
    psnr_db: float              # >40 dB ≈ imperceptible
    bbox: List[float]           # [x1, y1, x2, y2] in original image coords


@dataclass
class CloakResult:
    cloaked_image: Image.Image
    faces_found: int
    per_face_metrics: List[FaceMetrics] = field(default_factory=list)
    method: str = "mifgsm"
    epsilon_used: float = 8 / 255
    steps: int = 10
    warning: Optional[str] = None


# ── Normalisation helper ─────────────────────────────────────────────────────
def _prewhiten(t: torch.Tensor) -> torch.Tensor:
    """
    Per-image z-score standardisation used by InceptionResnetV1 in
    facenet-pytorch v0.1.0.  Equivalent to the package's `prewhiten()`:
        y = (x - mean) / max(std, 1/sqrt(n_elements))
    Accepts any shape tensor; operates element-wise.
    """
    mean = t.mean()
    std = t.std()
    std_adj = std.clamp(min=1.0 / (float(t.numel()) ** 0.5))
    return (t - mean) / std_adj


def _embed(resnet, t: torch.Tensor) -> torch.Tensor:
    """
    t  : float32 tensor in [0, 1], shape (1, 3, H, W)
    Returns 512-d L2-normalised embedding from InceptionResnetV1.
    We apply per-image prewhitening to match what InceptionResnetV1
    expects in facenet-pytorch v0.1.0.
    """
    return resnet(_prewhiten(t))


# ── Attack implementations ───────────────────────────────────────────────────

def _fgsm_cloak(
    x: torch.Tensor,
    clean_emb: torch.Tensor,
    resnet,
    epsilon: float,
) -> torch.Tensor:
    """
    Single-step FGSM (Goodfellow et al., 2014).
    x_adv = x - epsilon * sign(∇_x cosine_sim(embed(x), clean_emb))
    The negative sign descends cosine similarity (dodging direction).

    Uses torch.autograd.grad for explicit gradient computation (avoids any
    interaction with model parameter gradients that loss.backward() accumulates).
    """
    x_in = x.clone().detach().requires_grad_(True)
    loss = F.cosine_similarity(_embed(resnet, x_in), clean_emb).mean()
    grad = torch.autograd.grad(loss, x_in)[0]
    with torch.no_grad():
        x_adv = x_in - epsilon * grad.sign()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv.detach()


def _pgd_cloak(
    x: torch.Tensor,
    clean_emb: torch.Tensor,
    resnet,
    epsilon: float,
    alpha: float,
    steps: int,
) -> torch.Tensor:
    """
    Projected Gradient Descent (Madry et al., 2018) — iterative FGSM
    with L∞ projection back into the epsilon-ball each step.
    """
    x_adv = x.clone().detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = F.cosine_similarity(_embed(resnet, x_adv), clean_emb).mean()
        grad = torch.autograd.grad(loss, x_adv)[0]
        with torch.no_grad():
            x_adv = x_adv.detach() - alpha * grad.sign()
            # L∞ projection: keep within epsilon-ball of original x
            x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv.detach()


def _mifgsm_cloak(
    x: torch.Tensor,
    clean_emb: torch.Tensor,
    resnet,
    epsilon: float,
    alpha: float,
    steps: int,
    decay: float = 1.0,
) -> torch.Tensor:
    """
    Momentum Iterative FGSM (Dong et al., 2018 — "Boosting Adversarial
    Attacks with Momentum").  Accumulates a momentum term on the
    normalised gradient to improve transfer to unseen models.

    g_t = decay * g_{t-1} + grad / ||grad||_1
    x_adv = x_adv - alpha * sign(g_t)
    """
    x_adv = x.clone().detach()
    g = torch.zeros_like(x)  # momentum buffer
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = F.cosine_similarity(_embed(resnet, x_adv), clean_emb).mean()
        grad = torch.autograd.grad(loss, x_adv)[0]
        with torch.no_grad():
            # Normalise gradient by its L1 norm, accumulate momentum
            grad_norm = grad.abs().sum() + 1e-12
            g = decay * g + grad / grad_norm
            x_adv = x_adv.detach() - alpha * g.sign()
            # L∞ projection + clamp to valid image range
            x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv.detach()


# ── Per-crop cloaking helper ─────────────────────────────────────────────────

def _cloak_crop(
    crop_uint8: np.ndarray,
    method: str,
    epsilon: float,
    alpha: float,
    steps: int,
    decay: float,
) -> tuple[np.ndarray, float]:
    """
    Apply the chosen attack to a single 160×160 uint8 RGB crop.
    Returns (cloaked_uint8_crop, cosine_similarity_after).
    """
    # (H,W,C) uint8 → (1,C,H,W) float32 in [0,1]
    x = (
        torch.tensor(crop_uint8, dtype=torch.float32, device=_device)
        .permute(2, 0, 1)
        .unsqueeze(0)
        / 255.0
    )

    with torch.no_grad():
        clean_emb = _embed(_resnet, x).detach()

    if method == "fgsm":
        x_adv = _fgsm_cloak(x, clean_emb, _resnet, epsilon)
    elif method == "pgd":
        x_adv = _pgd_cloak(x, clean_emb, _resnet, epsilon, alpha, steps)
    else:  # mifgsm (default)
        x_adv = _mifgsm_cloak(x, clean_emb, _resnet, epsilon, alpha, steps, decay)

    with torch.no_grad():
        adv_emb = _embed(_resnet, x_adv).detach()
        cos_sim = F.cosine_similarity(adv_emb, clean_emb).item()

    adv_uint8 = (
        (x_adv.squeeze(0).permute(1, 2, 0) * 255)
        .round()
        .clamp(0, 255)
        .byte()
        .cpu()
        .numpy()
    )
    return adv_uint8, cos_sim


# ── Metrics helpers ──────────────────────────────────────────────────────────

def _compute_ssim_psnr(orig: np.ndarray, cloaked: np.ndarray) -> tuple[float, float]:
    """
    Returns (ssim, psnr_db) for uint8 RGB images.
    scikit-image >= 0.19 requires data_range to be set explicitly for
    floating-point inputs; we pass uint8 so data_range=255.
    """
    from skimage.metrics import (  # noqa: PLC0415
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


# ── Box → crop helper ────────────────────────────────────────────────────────

def _safe_crop_160(img_np: np.ndarray, box: list) -> Optional[np.ndarray]:
    """
    Crop a 160×160 region centred on the detected face bounding box,
    with margin=14 (matching MTCNN's own margin setting).
    Returns None if the box is degenerate.
    """
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    if x2 <= x1 or y2 <= y1:
        return None
    # expand with margin
    margin = 14
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    half = max((x2 - x1 + margin), (y2 - y1 + margin)) // 2
    H, W = img_np.shape[:2]
    lx = max(cx - half, 0)
    rx = min(cx + half, W)
    ly = max(cy - half, 0)
    ry = min(cy + half, H)
    patch = img_np[ly:ry, lx:rx]
    if patch.size == 0:
        return None
    # resize to exactly 160×160
    from PIL import Image as _PIL  # noqa: PLC0415
    patch_pil = _PIL.fromarray(patch)
    patch_pil = patch_pil.resize((160, 160), _PIL.BILINEAR)
    return np.array(patch_pil)


def _paste_crop_back(
    img_np: np.ndarray,
    cloaked_crop: np.ndarray,
    box: list,
) -> np.ndarray:
    """
    Resize the 160×160 cloaked crop back to the original face region
    and paste it into img_np (in place on a copy).
    """
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    margin = 14
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    half = max((x2 - x1 + margin), (y2 - y1 + margin)) // 2
    H, W = img_np.shape[:2]
    lx = max(cx - half, 0)
    rx = min(cx + half, W)
    ly = max(cy - half, 0)
    ry = min(cy + half, H)

    target_w, target_h = rx - lx, ry - ly
    if target_w <= 0 or target_h <= 0:
        return img_np

    from PIL import Image as _PIL  # noqa: PLC0415
    cloaked_pil = _PIL.fromarray(cloaked_crop).resize(
        (target_w, target_h), _PIL.BILINEAR
    )
    result = img_np.copy()
    result[ly:ry, lx:rx] = np.array(cloaked_pil)
    return result


# ── Main entry point ─────────────────────────────────────────────────────────

def cloak_image(
    pil_image: Image.Image,
    epsilon: float = 8 / 255,
    method: str = "mifgsm",
    steps: int = 10,
    alpha: float = 2 / 255,
    decay: float = 1.0,
) -> CloakResult:
    """
    Full cloaking pipeline:
      1. Detect all faces with MTCNN.
      2. For each face: extract 160×160 crop → apply attack → paste back.
      3. Compute per-face metrics (cosine_similarity, SSIM, PSNR).
      4. Return CloakResult.

    Parameters
    ----------
    pil_image : PIL Image (RGB)
    epsilon   : L∞ perturbation budget in [0,1] pixel space (default 8/255)
    method    : "fgsm" | "pgd" | "mifgsm"  (default "mifgsm")
    steps     : number of iterations for PGD/MI-FGSM (ignored for FGSM)
    alpha     : per-step size for iterative methods (default 2/255)
    decay     : momentum decay for MI-FGSM (default 1.0)
    """
    if not _models_loaded:
        raise RuntimeError(
            "Cloak models not loaded. Call load_models() before cloak_image()."
        )

    method = method.lower().strip()
    if method not in ("fgsm", "pgd", "mifgsm"):
        raise ValueError(f"Unknown method '{method}'. Choose fgsm | pgd | mifgsm.")

    img_rgb = pil_image.convert("RGB")
    img_np = np.array(img_rgb)

    # ── Face detection ───────────────────────────────────────────────────────
    boxes, probs = _mtcnn.detect(img_rgb)

    if boxes is None or len(boxes) == 0:
        logger.info("[cloak] No faces detected — returning original image.")
        return CloakResult(
            cloaked_image=img_rgb,
            faces_found=0,
            method=method,
            epsilon_used=epsilon,
            steps=steps,
            warning=(
                "No faces were detected in the image. "
                "The original image is returned unmodified. "
                "Ensure the image contains a clearly visible face."
            ),
        )

    logger.info(
        "[cloak] %d face(s) detected. method=%s, eps=%.4f, steps=%d",
        len(boxes),
        method,
        epsilon,
        steps,
    )

    # ── Per-face cloaking ────────────────────────────────────────────────────
    cloaked_np = img_np.copy()
    per_face_metrics: List[FaceMetrics] = []

    for idx, (box, prob) in enumerate(zip(boxes, probs)):
        logger.debug("[cloak] Face %d  prob=%.3f  box=%s", idx, prob or 0.0, box)

        crop = _safe_crop_160(img_np, box)
        if crop is None:
            logger.warning("[cloak] Face %d: degenerate crop — skipping.", idx)
            continue

        try:
            adv_crop, cos_sim = _cloak_crop(
                crop_uint8=crop,
                method=method,
                epsilon=epsilon,
                alpha=alpha,
                steps=steps,
                decay=decay,
            )
        except Exception as exc:
            logger.exception("[cloak] Face %d attack failed: %s", idx, exc)
            continue

        # Paste adversarial crop back
        cloaked_np = _paste_crop_back(cloaked_np, adv_crop, box)

        # Metrics on the 160×160 crop only (fast, representative)
        ssim_val, psnr_val = _compute_ssim_psnr(crop, adv_crop)
        per_face_metrics.append(
            FaceMetrics(
                face_idx=idx,
                cosine_similarity=round(cos_sim, 5),
                ssim=round(ssim_val, 5),
                psnr_db=round(psnr_val, 3),
                bbox=[round(float(v), 1) for v in box],
            )
        )
        logger.info(
            "[cloak] Face %d  cos_sim=%.4f  SSIM=%.4f  PSNR=%.1f dB",
            idx, cos_sim, ssim_val, psnr_val,
        )

    if not per_face_metrics:
        return CloakResult(
            cloaked_image=img_rgb,
            faces_found=len(boxes),
            method=method,
            epsilon_used=epsilon,
            steps=steps,
            warning="Faces were detected but all crops failed during the attack step.",
        )

    cloaked_pil = Image.fromarray(cloaked_np)
    return CloakResult(
        cloaked_image=cloaked_pil,
        faces_found=len(boxes),
        per_face_metrics=per_face_metrics,
        method=method,
        epsilon_used=epsilon,
        steps=steps,
        warning=None,
    )
