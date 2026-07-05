"""
app/identifier/identifier_service.py
=====================================
CLIP-based zero-shot image classification.

Surrogate model : CLIP ViT-B/32 (reused from RAG/embedder.py — no double load)
Labels          : 20 generic scene/object categories
Output          : top label + full ranked score list (softmax probabilities)

The model's learned logit_scale temperature is applied before softmax so that
the score distribution is sharp and decisive (matches CLIP's original forward()).
"""

from __future__ import annotations

import logging
from typing import Any

import clip
import torch
import torch.nn.functional as F
from PIL import Image

from RAG.embedder import device as _device
from RAG.embedder import model as _clip_model

logger = logging.getLogger(__name__)

# ── CLIP normalization constants (ViT-B/32) ──────────────────────────────────
_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=_device).view(1, 3, 1, 1)
_CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=_device).view(1, 3, 1, 1)

# ── Label definitions ─────────────────────────────────────────────────────────
LABELS: list[str] = [
    "a photo of a person",
    "a photo of a dog",
    "a photo of a cat",
    "a photo of a car",
    "a photo of food",
    "a photo of a building",
    "a photo of a landscape",
    "a photo of a bird",
    "a photo of a tree",
    "a photo of a bicycle",
    "a photo of furniture",
    "a photo of text or a document",
    "a photo of an animal",
    "a photo of sports or exercise",
    "a photo of an indoor scene",
    "a photo of an outdoor scene",
    "a photo of art or a painting",
    "a photo of technology or electronics",
    "a photo of a vehicle",
    "a photo of something abstract or unrecognizable",
]

SHORT_LABELS: dict[str, str] = {
    "a photo of a person": "Person",
    "a photo of a dog": "Dog",
    "a photo of a cat": "Cat",
    "a photo of a car": "Car",
    "a photo of food": "Food",
    "a photo of a building": "Building",
    "a photo of a landscape": "Landscape",
    "a photo of a bird": "Bird",
    "a photo of a tree": "Tree",
    "a photo of a bicycle": "Bicycle",
    "a photo of furniture": "Furniture",
    "a photo of text or a document": "Text / Document",
    "a photo of an animal": "Animal",
    "a photo of sports or exercise": "Sports",
    "a photo of an indoor scene": "Indoor Scene",
    "a photo of an outdoor scene": "Outdoor Scene",
    "a photo of art or a painting": "Art / Painting",
    "a photo of technology or electronics": "Technology",
    "a photo of a vehicle": "Vehicle",
    "a photo of something abstract or unrecognizable": "Unrecognizable",
}

# ── Pre-compute text embeddings once at import time ───────────────────────────
_text_embs: torch.Tensor  # (20, 512)

def _build_text_embeddings() -> torch.Tensor:
    with torch.no_grad():
        tokens = clip.tokenize(LABELS).to(_device)
        embs = _clip_model.encode_text(tokens)
        return F.normalize(embs, dim=-1)

_text_embs_loaded = False
try:
    _text_embs = _build_text_embeddings()
    _text_embs_loaded = True
    logger.info("[identifier] Text embeddings pre-computed for %d labels.", len(LABELS))
except Exception as _e:
    logger.error("[identifier] Failed to pre-compute text embeddings: %s", _e)
    _text_embs = None  # type: ignore[assignment]


def get_status() -> dict[str, Any]:
    return {
        "loaded": _text_embs_loaded,
        "device": str(_device),
        "surrogate": "CLIP-ViT-B/32",
        "labels_count": len(LABELS),
    }


def classify_image(pil_image: Image.Image) -> dict[str, Any]:
    """
    Run CLIP zero-shot classification against the 20 predefined labels.

    Returns
    -------
    {
        "top_label"      : str,
        "top_label_short": str,
        "confidence"     : float,   # 0–1, softmax probability
        "all_scores"     : [{"label": str, "short": str, "score": float}, ...]
                           # sorted descending by score
    }
    """
    if not _text_embs_loaded:
        raise RuntimeError("Text embeddings not loaded — check server logs.")

    img_rgb = pil_image.convert("RGB")
    img_224 = img_rgb.resize((224, 224), Image.LANCZOS)

    # (H, W, C) → (1, C, H, W) float32 in [0, 1]
    import numpy as np
    arr = np.array(img_224).astype("float32") / 255.0
    x = torch.tensor(arr, device=_device).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        # Apply CLIP normalization
        x_norm = (x - _CLIP_MEAN) / _CLIP_STD
        img_emb = F.normalize(_clip_model.encode_image(x_norm), dim=-1)  # (1, 512)

        # Cosine similarities scaled by the model's learned temperature
        logit_scale = _clip_model.logit_scale.exp()
        scores = logit_scale * (img_emb @ _text_embs.T)  # (1, 20)
        probs = torch.softmax(scores, dim=-1).squeeze(0)  # (20,)

    # Sort descending
    sorted_idx = probs.argsort(descending=True).cpu().tolist()
    all_scores = [
        {
            "label": LABELS[i],
            "short": SHORT_LABELS[LABELS[i]],
            "score": round(float(probs[i]), 5),
        }
        for i in sorted_idx
    ]

    top = all_scores[0]
    logger.info(
        "[identifier] Top: '%s'  confidence=%.3f",
        top["short"], top["score"],
    )

    return {
        "top_label": top["label"],
        "top_label_short": top["short"],
        "confidence": top["score"],
        "all_scores": all_scores,
    }
