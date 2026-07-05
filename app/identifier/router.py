"""
POST /identifier
  Multipart form:
    image : image file (required)
    force : optional short-label override (testing)

  Response JSON:
    {
      "top_label"      : str,
      "top_label_short": str,
      "confidence"     : float,
      "all_scores"     : [{"label": str, "short": str, "score": float}, ...]
    }

GET /identifier/health — CLIP model load status.
"""
from __future__ import annotations

import io
import logging

from fastapi import APIRouter, File, Form, Query, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from app.identifier.identifier_service import classify_image, get_status, _text_embs_loaded

logger = logging.getLogger(__name__)

router = APIRouter()

_MAX_IMAGE_BYTES = 20 * 1024 * 1024  # 20 MB


@router.get("/health")
def health():
    return get_status()


@router.post("")
def identify(
    image: UploadFile | None = File(None),
    force: str | None = Form(None),
    force_query: str | None = Query(None, alias="force"),
):
    force = force_query or force
    if not _text_embs_loaded:
        return JSONResponse(status_code=503, content={"error": "Identifier model not loaded. Check /identifier/health."})

    if image is None:
        return JSONResponse(status_code=400, content={"error": "Missing 'image' file in multipart form."})

    raw_bytes = image.file.read()

    if len(raw_bytes) == 0:
        return JSONResponse(status_code=400, content={"error": "Uploaded file is empty."})
    if len(raw_bytes) > _MAX_IMAGE_BYTES:
        return JSONResponse(status_code=413, content={"error": f"File too large (max {_MAX_IMAGE_BYTES // 1024 // 1024} MB)."})

    try:
        pil_image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Could not decode image. Send a valid JPG/PNG/WEBP."})

    if force:
        from app.identifier.identifier_service import LABELS, SHORT_LABELS
        label = next((l for l in LABELS if SHORT_LABELS[l].lower() == force.lower()), None)
        if label is None:
            label = f"a photo of {force.lower()}"
            short = force
        else:
            short = SHORT_LABELS[label]
        fake_scores = [{"label": label, "short": short, "score": 0.99}] + [
            {"label": l, "short": SHORT_LABELS.get(l, l), "score": round(0.01 / max(len(LABELS) - 1, 1), 5)}
            for l in LABELS if l != label
        ]
        return {"top_label": label, "top_label_short": short, "confidence": 0.99, "all_scores": fake_scores}

    try:
        result = classify_image(pil_image)
    except Exception:
        logger.exception("[identifier] Classification failed.")
        return JSONResponse(status_code=500, content={"error": "Internal classification error. See server logs."})

    return result
