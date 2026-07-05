"""
POST /cloak
  Multipart form:
    image    : image file  (required)
    mode     : str         (optional, "auto" | "face" | "clip" | "dual", default "auto")
    epsilon  : float       (optional, default 0.03137 ≈ 8/255)
    method   : str         (optional, "fgsm" | "pgd" | "mifgsm", default "mifgsm")
    steps    : int         (optional, default 10; ignored for fgsm)
    alpha    : float       (optional, default 0.00784 ≈ 2/255)
    decay    : float       (optional, default 1.0; only for mifgsm)
    target_prompt : str    (optional, CLIP shield target)

  Response JSON: protection_mode/protection_target/faces_found/
  cloaked_image_base64/noise_heatmap_base64/metrics{face.per_face[], clip{},
  epsilon_used, method, steps}/warning.

GET /cloak/health — model load status and device.
"""
from __future__ import annotations

import base64
import io
import logging

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from app.cloak.cloak_service import (
    cloak_image,
    get_model_status,
    load_models,
    get_mtcnn,
)
from app.cloak.clip_service import (
    shield_image,
    get_clip_status,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# ── Load models eagerly at import time (not per-request) ─────────────────────
try:
    load_models()
except Exception as _load_exc:
    logger.error(
        "[cloak] Model load at import time failed — endpoint will return 503: %s",
        _load_exc,
    )

_ALLOWED_METHODS = {"fgsm", "pgd", "mifgsm"}
_MAX_IMAGE_BYTES = 20 * 1024 * 1024  # 20 MB safety cap


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pil_to_base64_png(img: Image.Image) -> str:
    """Encode a PIL Image as a PNG and return the base64 string."""
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _parse_float(value: str | None, default: float, lo: float, hi: float) -> float:
    if value is None:
        return default
    try:
        v = float(value)
    except ValueError:
        return default
    return max(lo, min(hi, v))


def _parse_int(value: str | None, default: int, lo: int, hi: int) -> int:
    if value is None:
        return default
    try:
        v = int(value)
    except ValueError:
        return default
    return max(lo, min(hi, v))


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/health")
def health():
    """Return model load status and device info."""
    status = get_model_status()
    clip_status = get_clip_status()
    http_code = 200 if status["loaded"] and clip_status["loaded"] else 503
    return JSONResponse(status_code=http_code, content={"face": status, "clip": clip_status})


@router.post("")
def cloak(
    image: UploadFile | None = File(None),
    mode: str | None = Form(None),
    target_prompt: str | None = Form(None),
    method: str | None = Form(None),
    epsilon: str | None = Form(None),
    steps: str | None = Form(None),
    alpha: str | None = Form(None),
    decay: str | None = Form(None),
):
    """
    Apply adversarial face cloaking to the uploaded image.
    All parameters are read from the multipart form body.
    """
    status = get_model_status()
    if not status["loaded"]:
        return JSONResponse(
            status_code=503,
            content={"error": "Cloaking models are not loaded. Check /cloak/health."},
        )

    # ── Image file ────────────────────────────────────────────────────────────
    if image is None:
        return JSONResponse(status_code=400, content={"error": "Missing 'image' file in multipart form."})

    raw_bytes = image.file.read()
    if len(raw_bytes) == 0:
        return JSONResponse(status_code=400, content={"error": "Uploaded file is empty."})
    if len(raw_bytes) > _MAX_IMAGE_BYTES:
        return JSONResponse(
            status_code=413,
            content={"error": f"File too large (max {_MAX_IMAGE_BYTES // 1024 // 1024} MB)."},
        )

    try:
        pil_image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Could not decode image. Send a valid JPG/PNG/WEBP."})

    # ── Attack parameters ─────────────────────────────────────────────────────
    mode = (mode or "auto").lower().strip()
    if mode not in ("auto", "face", "clip", "dual"):
        return JSONResponse(status_code=400, content={"error": f"Invalid mode '{mode}'."})

    if target_prompt is not None:
        target_prompt = target_prompt.strip() or None

    method = (method or "mifgsm").lower().strip()
    if method not in _ALLOWED_METHODS:
        return JSONResponse(
            status_code=400,
            content={"error": f"Invalid method '{method}'. Choose: fgsm | pgd | mifgsm."},
        )

    epsilon = _parse_float(epsilon, default=8 / 255, lo=1 / 255, hi=32 / 255)
    steps = _parse_int(steps, default=10, lo=1, hi=100)
    alpha = _parse_float(alpha, default=2 / 255, lo=0.5 / 255, hi=epsilon)  # alpha <= epsilon
    decay = _parse_float(decay, default=1.0, lo=0.0, hi=1.0)

    logger.info(
        "[cloak] Request: mode=%s method=%s eps=%.5f steps=%d alpha=%.5f decay=%.2f  "
        "image_size=%dx%d",
        mode, method, epsilon, steps, alpha, decay, pil_image.width, pil_image.height,
    )

    # ── Routing ───────────────────────────────────────────────────────────────
    mtcnn = get_mtcnn()

    if mode == "auto":
        boxes, _ = mtcnn.detect(pil_image)
        # Always run the CLIP attack so the identifier can't recognise the image;
        # add the face attack on top when human faces are present.
        mode = "dual" if (boxes is not None and len(boxes) > 0) else "clip"

    try:
        face_result = None
        clip_result = None
        final_image = pil_image

        if mode in ("face", "dual"):
            face_result = cloak_image(
                pil_image=final_image,
                epsilon=epsilon,
                method=method,
                steps=steps,
                alpha=alpha,
                decay=decay,
            )
            final_image = face_result.cloaked_image

        if mode in ("clip", "dual"):
            clip_result = shield_image(
                pil_image=final_image,
                epsilon=epsilon,
                method=method,
                steps=steps,
                alpha=alpha,
                decay=decay,
                target_prompt=target_prompt,
            )
            final_image = clip_result.cloaked_image
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    except Exception:
        logger.exception("[cloak] Unhandled error during cloaking.")
        return JSONResponse(status_code=500, content={"error": "Internal cloaking error. See server logs."})

    # ── Encode response ───────────────────────────────────────────────────────
    try:
        b64 = _pil_to_base64_png(final_image)
        noise_b64 = None
        if clip_result and clip_result.noise_heatmap:
            noise_b64 = _pil_to_base64_png(clip_result.noise_heatmap)
    except Exception:
        logger.exception("[cloak] Failed to encode result image.")
        return JSONResponse(status_code=500, content={"error": "Failed to encode output image."})

    response_body = {
        "protection_mode": mode,
        "protection_target": "FaceNet/ArcFace" if mode == "face" else "CLIP training scrapers" if mode == "clip" else "Both FaceNet and CLIP",
        "faces_found": face_result.faces_found if face_result else 0,
        "cloaked_image_base64": b64,
        "noise_heatmap_base64": noise_b64,
        "metrics": {
            "face": {
                "per_face": [
                    {
                        "face_idx": m.face_idx,
                        "cosine_similarity": m.cosine_similarity,
                        "ssim": m.ssim,
                        "psnr_db": m.psnr_db,
                        "bbox": m.bbox,
                    }
                    for m in face_result.per_face_metrics
                ]
            } if face_result else None,
            "clip": {
                "cosine_sim_before": clip_result.cosine_sim_before,
                "cosine_sim_after": clip_result.cosine_sim_after,
                "semantic_shift_pct": clip_result.semantic_shift_pct,
                "ssim": clip_result.ssim,
                "psnr_db": clip_result.psnr_db,
                "target_prompt": clip_result.target_prompt,
            } if clip_result else None,
            "epsilon_used": round(epsilon, 6),
            "method": method,
            "steps": steps,
        },
        "warning": (face_result.warning if face_result else None) or (clip_result.warning if clip_result else None),
    }

    logger.info("[cloak] Done. mode=%s method=%s", mode, method)
    return response_body
