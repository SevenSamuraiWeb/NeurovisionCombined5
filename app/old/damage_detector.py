"""Robust damage mask detection for old photos.

The trained segmentation head in `OldPhotoDamageModel` overfits to synthetic
training data and collapses to predicting damage everywhere on out-of-distribution
real photos. This module provides a pretrained-free, classical-CV-based detector
that mirrors the preprocessing pipeline used in Microsoft's "Bringing Old Photos
Back to Life" and the broader restoration literature:

  - **Scratches** are thin, curvilinear, low-curvature structures that are
    locally brighter or darker than their surround. Two detectors run in
    parallel and are unioned:

      a. Frangi vesselness (Frangi et al., 1998): analyses Hessian eigenvalues
         to score how "tube-like" each pixel is. We run on the image and its
         inverse to catch bright and dark scratches alike.

      b. Multi-scale morphological top-hat / bottom-hat: finds pixels that are
         meaningfully brighter (or darker) than a large morphological background
         estimate. Catches high-contrast cracks that have a simpler cross-section
         than the Frangi shape filter assumes.

  - **Missing patches** are large connected regions of near-pure white or
    near-pure black (paper torn off, burn marks, dropouts). Detected via
    extreme-value thresholding + connected-component area filtering.

  - **Stains** show up as smooth chromaticity outliers — local color deviation
    from a Gaussian-blurred neighborhood in LAB space.

The trained seg model is used as a corroborating signal when it produces a
plausible mask (damage_ratio in [0.01, 0.5]); otherwise it is discarded.

IMPORTANT: scratch and missing-patch detectors always run regardless of the
classifier's predicted_types. The classifier frequently omits 'scratches' on
real OOD photos even when physical damage is visually obvious.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from skimage.filters import frangi
from skimage.morphology import remove_small_objects, binary_closing, disk

_BOPBTL_PATH = Path(__file__).resolve().parents[2] / "models" / "bopbtl_scratch.pt"

class _BopbtlScratchDetector:
    """Neural scratch detector from BOPBTL (CVPR 2020). Lazy singleton.

    Loads on first use; subsequent calls reuse the same loaded model.
    Falls back to None (classical Frangi only) if weights are unavailable or
    loading fails for any reason.
    """

    _instance: "Optional[_BopbtlScratchDetector]" = None
    _init_attempted: bool = False
    _lock = threading.Lock()

    @classmethod
    def get(cls) -> "Optional[_BopbtlScratchDetector]":
        # Double-checked locking: the constructor releases the GIL during the
        # multi-second GPU model load, so without the lock a concurrent caller
        # could see `_init_attempted=True` and read a still-None `_instance`,
        # silently falling back to classical-only detection. `_init_attempted`
        # is set only AFTER construction completes, so concurrent callers block
        # on the lock until the first load finishes, then see the real instance.
        if cls._init_attempted:
            return cls._instance
        with cls._lock:
            if cls._init_attempted:
                return cls._instance
            try:
                cls._instance = cls()
            except Exception as exc:
                print(f"[damage_detector] BOPBTL init failed, using classical only: {exc}", flush=True)
            finally:
                cls._init_attempted = True
        return cls._instance

    def __init__(self) -> None:
        import torch
        from app.old.vendored.bopbtl.detection_networks import UNet

        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        if not _BOPBTL_PATH.exists():
            raise FileNotFoundError(
                f"BOPBTL scratch detection weights not found at {_BOPBTL_PATH}. "
                "Download global_checkpoints.zip from the BOPBTL v1.0 release, "
                "extract Global/checkpoints/detection/FT_Epoch_latest.pt, "
                f"and place it at {_BOPBTL_PATH}."
            )

        # Architecture must match Global/detection.py exactly.
        self._model = UNet(
            in_channels=1, out_channels=1, depth=4, conv_num=2, wf=6,
            padding=True, batch_norm=True, up_mode="upsample",
            with_tanh=False, antialiasing=True,
        )

        ckpt = torch.load(str(_BOPBTL_PATH), map_location="cpu", weights_only=False)
        if not isinstance(ckpt, dict) or "model_state" not in ckpt:
            raise RuntimeError(
                f"Unexpected BOPBTL checkpoint at {_BOPBTL_PATH}: "
                "expected dict with key 'model_state' (FT_Epoch_latest.pt)."
            )
        state = ckpt["model_state"]
        # Upstream wraps the UNet in DataParallelWithCallback when sync_bn=True,
        # so saved keys are prefixed with "module.". We build the UNet without
        # the wrapper, so strip the prefix.
        state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
        self._model.load_state_dict(state, strict=True)
        self._model.to(self._device)
        self._model.eval()
        print("[damage_detector] BOPBTL scratch detector loaded.", flush=True)

    def detect(self, image: Image.Image) -> np.ndarray:
        """Return uint8 mask (0/255) at the same H×W as `image`."""
        import torch

        orig_w, orig_h = image.size

        # Mirror Global/detection.py data_transforms("scale_256"): short edge
        # 256, long edge proportional, both rounded to a multiple of 16.
        if orig_w < orig_h:
            new_w = 256
            new_h = int(round(orig_h / orig_w * 256))
        else:
            new_h = 256
            new_w = int(round(orig_w / orig_h * 256))
        new_w = max(16, int(round(new_w / 16) * 16))
        new_h = max(16, int(round(new_h / 16) * 16))
        resized = image.resize((new_w, new_h), Image.BICUBIC).convert("L")

        gray = np.asarray(resized, dtype=np.float32) / 255.0
        gray = (gray - 0.5) / 0.5  # [-1, 1]
        tensor = torch.from_numpy(gray)[None, None, :, :].to(self._device)

        with torch.no_grad():
            prob = torch.sigmoid(self._model(tensor))  # (1,1,H,W) ∈ [0,1]

        mask_small = (prob.squeeze().cpu().numpy() >= 0.4).astype(np.uint8) * 255
        if (new_w, new_h) != (orig_w, orig_h):
            mask_small = np.array(
                Image.fromarray(mask_small).resize((orig_w, orig_h), Image.NEAREST)
            )
        return mask_small


def _detect_scratches_neural(image_pil: Image.Image) -> "Optional[np.ndarray]":
    """Try BOPBTL neural scratch detection. Returns uint8 mask (0/255) or None on failure."""
    detector = _BopbtlScratchDetector.get()
    if detector is None:
        return None
    try:
        return detector.detect(image_pil)
    except Exception as exc:
        print(f"[damage_detector] BOPBTL detection failed: {exc}", flush=True)
        return None


@dataclass
class DamageDetectionResult:
    mask: np.ndarray  # uint8, 0 or 255, same shape as input image (H, W)
    components: dict[str, np.ndarray]  # individual contributing masks
    source: str  # "classical", "classical+model", or "model_only"
    coverage: float  # fraction of pixels marked damaged


def _detect_scratches(gray: np.ndarray) -> np.ndarray:  # noqa: ARG001  (kept-for-debug)
    """Deprecated. Frangi/top-hat scratch detector — false-fires on facial
    features (eye sockets, nose) at low resolution, so it's no longer used
    by `detect_damage_mask`. Kept here only for offline debugging.
    """
    norm = gray.astype(np.float32) / 255.0
    sigmas = [0.8, 1.2, 1.8, 2.6, 4.0, 6.0, 8.0]
    bright = frangi(norm, sigmas=sigmas, black_ridges=False)  # type: ignore[arg-type]
    dark = frangi(norm, sigmas=sigmas, black_ridges=True)  # type: ignore[arg-type]
    combined = np.maximum(bright, dark)

    min_obj = max(25, gray.size // 5000)

    # ---- Frangi thresholds ----
    frangi_conservative = np.zeros(gray.shape, dtype=bool)
    frangi_moderate = np.zeros(gray.shape, dtype=bool)
    if combined.max() > 1e-6:
        nonzero = combined[combined > 0]
        cut_96 = float(np.quantile(nonzero, 0.96)) if nonzero.size else 1.0
        cut_90 = float(np.quantile(nonzero, 0.90)) if nonzero.size else 1.0
        frangi_conservative = combined >= max(cut_96, 0.015)
        frangi_moderate = combined >= max(cut_90, 0.008)

    # ---- Multi-scale top-hat / bottom-hat ----
    tophat = np.zeros(gray.shape, dtype=np.float32)
    bothat = np.zeros(gray.shape, dtype=np.float32)
    gray_f = gray.astype(np.float32)
    for ks in (15, 25, 37, 51):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        tophat = np.maximum(tophat, cv2.morphologyEx(gray_f, cv2.MORPH_TOPHAT, k))
        bothat = np.maximum(bothat, cv2.morphologyEx(gray_f, cv2.MORPH_BLACKHAT, k))
    morpho = np.maximum(tophat, bothat)

    morpho_extreme = np.zeros(gray.shape, dtype=bool)
    morpho_moderate = np.zeros(gray.shape, dtype=bool)
    if morpho.max() > 0:
        mu, sigma = morpho.mean(), morpho.std()
        # "Extreme": 4.5σ above mean or at least 35 gray levels — catches only
        # pure-white torn paper / very high-contrast folds.
        morpho_extreme = morpho >= max(mu + 4.5 * sigma, 35.0)
        # "Moderate": 3.0σ, used only in conjunction with Frangi (Path C).
        morpho_moderate = morpho >= max(mu + 3.0 * sigma, 22.0)

    def _clean(m: np.ndarray) -> np.ndarray:
        if not m.any():
            return m
        m = binary_closing(m, disk(2))
        return remove_small_objects(m, min_size=min_obj)

    path_a = _clean(frangi_conservative)           # Frangi alone, conservative
    path_b = _clean(morpho_extreme)                 # Top-hat alone, very strict
    path_c = _clean(frangi_moderate & morpho_moderate)  # Both agree, moderate

    return path_a | path_b | path_c


def _detect_missing_patches(gray: np.ndarray) -> np.ndarray:
    """Compact regions of saturated white/black that are LOCAL outliers — torn or dropout areas.

    Two guards distinguish damage from a naturally light/dark background:
      1. The region must be brighter (or darker) than its local neighborhood
         by a margin, not just above a global threshold. We compare each pixel
         to a large Gaussian-blurred local mean.
      2. After connected-component cleaning, any single component covering
         more than 25% of the image is discarded — true torn regions don't
         dominate the frame; if a giant blob shows up, it's the background.
    """
    h, w = gray.shape
    min_area = max(64, (h * w) // 4000)
    max_component_frac = 0.25

    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(8.0, min(h, w) / 16.0))
    delta = gray.astype(np.int16) - blurred.astype(np.int16)
    # Lowered bright threshold 245 → 238 and delta 30 → 25:
    # old photo torn edges frequently read as 238-244, not full 255.
    white = (gray > 238) & (delta > 25)
    black = (gray < 10) & (delta < -30)
    raw = white | black
    if not raw.any():
        return np.zeros_like(gray, dtype=bool)
    cleaned = remove_small_objects(raw, min_size=min_area)
    cleaned = binary_closing(cleaned, disk(2))

    # Discard runaway components (the dominant background flagged as a single blob).
    n_labels, labels = cv2.connectedComponents(cleaned.astype(np.uint8))
    max_pixels = max_component_frac * h * w
    out = np.zeros_like(cleaned)
    for label_id in range(1, n_labels):
        component = labels == label_id
        if component.sum() <= max_pixels:
            out |= component
    return out


def _detect_stains(bgr: np.ndarray) -> np.ndarray:
    """Chromaticity outliers in LAB — pixels far from their local color neighborhood.

    Best-effort: stains are the least clean classical signal. We blur the
    a/b channels and flag pixels whose chromaticity deviates strongly from the
    blurred local mean."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    ab = lab[..., 1:].astype(np.float32)
    blurred = cv2.GaussianBlur(ab, (0, 0), sigmaX=25)
    delta = np.linalg.norm(ab - blurred, axis=2)
    if delta.max() < 1e-6:
        return np.zeros(bgr.shape[:2], dtype=bool)
    # 99.5th percentile — only the most extreme deviations
    cutoff = np.quantile(delta, 0.995)
    mask = delta >= max(cutoff, 12.0)
    mask = remove_small_objects(mask, min_size=64)
    return mask


def detect_damage_mask(
    image: Image.Image,
    model_mask: np.ndarray | None = None,
    model_damage_ratio: float | None = None,
    predicted_types: list[str] | None = None,
    sam_expander=None,
) -> DamageDetectionResult:
    """Build a robust damage mask combining classical detectors and (optionally) the trained model.

    Args:
        image: PIL RGB image at any resolution.
        model_mask: Optional uint8 mask from the trained seg head (any shape, will be resized).
        model_damage_ratio: Optional damage_ratio reported by the model.
        predicted_types: Optional classifier output — used only to gate the stain detector.

    Returns:
        DamageDetectionResult with `mask` at the same H×W as the input image.
    """
    arr_rgb = np.array(image.convert("RGB"))
    bgr = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    types = set(predicted_types or [])

    # Scratch and missing-patch detectors always run — the classifier frequently
    # omits 'scratches' on real OOD photos even when physical damage is obvious.
    # The coverage > 0.25 prune below handles over-detection on clean images.
    run_scratches = True
    run_missing = True
    # Stain detection is colour-dependent; only run when classifier flagged it.
    run_stains = "stains" in types

    components: dict[str, np.ndarray] = {}
    if run_scratches:
        # BOPBTL is the sole scratch signal. The classical Frangi/top-hat
        # detector was a fallback for when BOPBTL was broken, but it
        # systematically false-fires on facial features (eye sockets, nose
        # contour) at low resolutions, where Frangi vesselness reads them
        # as curvilinear ridges. BOPBTL was trained for this task.
        neural_u8 = _detect_scratches_neural(image)
        if neural_u8 is not None:
            neural_bool = neural_u8 > 0
            # Drop if it fires on > 25% of the image (runaway detection).
            if float(neural_bool.mean()) <= 0.25:
                components["scratches"] = neural_bool
        # If BOPBTL is unavailable or runaway, leave scratches absent — better
        # no inpainting than over-inpainting based on face contours.
    if run_missing:
        components["missing_patch"] = _detect_missing_patches(gray)
    if run_stains:
        components["stains"] = _detect_stains(bgr)

    # Any individual component exceeding 25% coverage is not damage. Drop it.
    pruned: dict[str, np.ndarray] = {}
    for name, m in components.items():
        if float(m.mean()) > 0.25:
            continue
        pruned[name] = m
    components = pruned

    classical = np.zeros((h, w), dtype=bool)
    for m in components.values():
        classical |= m

    source = "classical"
    final = classical

    # Corroborate with the trained model only if it produced a plausible (not
    # collapsed) mask. The model collapses to damage_ratio ≈ 1.0 on OOD inputs.
    if model_mask is not None and model_damage_ratio is not None and 0.01 <= model_damage_ratio <= 0.5:
        if model_mask.shape != (h, w):
            model_mask_resized = np.array(
                Image.fromarray(model_mask).resize((w, h), Image.NEAREST)
            )
        else:
            model_mask_resized = model_mask
        model_bool = model_mask_resized >= 128
        if classical.any():
            # Keep model pixels that corroborate the classical mask (overlap its
            # 8px dilation)...
            corroborated = model_bool & binary_closing(classical, disk(8))
            # ...PLUS standalone model components that are large enough to be real
            # damage rather than speckle. Previously the model mask was AND'd down
            # to the classical mask, silently discarding model-detected damage in
            # regions the classical detectors missed entirely (e.g. a faded stain
            # far from any scratch). The model mask has already passed the
            # 0.01–0.5 damage_ratio plausibility gate, so its sizeable components
            # are trustworthy; the per-component area band filters speckle (below
            # min_model_area) and runaway blobs (above the 25% cap).
            min_model_area = max(64, (h * w) // 4000)
            max_model_area = 0.25 * h * w
            standalone = np.zeros_like(model_bool)
            n_labels, labels = cv2.connectedComponents(model_bool.astype(np.uint8))
            for label_id in range(1, n_labels):
                comp = labels == label_id
                comp_area = int(comp.sum())
                if min_model_area <= comp_area <= max_model_area:
                    standalone |= comp
            final = classical | corroborated | standalone
            source = "classical+model"
        else:
            final = model_bool
            source = "model_only"

    mask_u8 = (final.astype(np.uint8)) * 255
    coverage = float(final.mean())

    if sam_expander is not None and coverage > 0:
        try:
            expanded_u8 = sam_expander.expand(image, mask_u8)
            expanded_bool = expanded_u8 > 0
            expanded_coverage = float(expanded_bool.mean())
            # SAM2 can return whole-object segmentations (e.g. the entire
            # background) when a centroid prompt lands on a long connected
            # scratch network. Reject if the expansion blows past 25%.
            if expanded_bool.sum() > final.sum() and expanded_coverage <= 0.25:
                mask_u8 = expanded_u8
                coverage = expanded_coverage
                source = source + "+sam"
            elif expanded_coverage > 0.25:
                print(
                    f"[damage_detector] SAM expansion rejected: coverage "
                    f"{expanded_coverage:.2f} > 0.25 (runaway segmentation).",
                    flush=True,
                )
        except Exception as exc:
            print(f"[damage_detector] SAM expansion failed, using classical mask: {exc}", flush=True)

    return DamageDetectionResult(
        mask=mask_u8,
        components={k: (v.astype(np.uint8) * 255) for k, v in components.items()},
        source=source,
        coverage=coverage,
    )
