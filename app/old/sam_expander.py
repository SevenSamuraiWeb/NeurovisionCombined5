"""SAM2-based mask expander for old photo damage detection.

Takes the classical damage mask (thin Frangi + morphological responses) as seed
signal. For each connected component in that mask, a SAM2 prediction is run using
the component centroid as a positive point prompt. The *smallest* SAM2 segment
whose overlap fraction with the classical mask meets `overlap_threshold` is merged
into the result. This converts thin Frangi scratch-line detections (1-3 px wide)
into the full crack-region masks (5-15 px wide, or larger tear areas).

Face and background regions are excluded naturally: their SAM2 segments are large
smooth blobs with very low overlap relative to thin Frangi lines, so they fall
below the threshold without needing explicit negative prompts.

SAM2 vs SAM1: same predict() signature, ~6x better boundary accuracy, weights
auto-downloaded from HuggingFace (facebook/sam2-hiera-base-plus, ~320 MB).
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from PIL import Image


def sam_available() -> bool:
    """Return True if sam2 is importable."""
    try:
        import sam2  # noqa: F401
        return True
    except ImportError:
        return False


class SamMaskExpander:
    """Expand a classical damage mask using SAM2 (Segment Anything Model v2).

    Uses SAM2ImagePredictor: one image encoding (expensive ViT forward pass) then
    one cheap decode per connected component — typically 15-25 components → ~1-2 s
    total after the initial set_image call.
    """

    def __init__(
        self,
        model_id: str = "facebook/sam2-hiera-base-plus",
        device: Optional[str] = None,
        overlap_threshold: float = 0.10,
        top_n_components: int = 20,
    ) -> None:
        """
        Args:
            model_id: HuggingFace model ID for SAM2. Weights auto-downloaded.
            device: torch device string; None → CUDA if available else CPU.
            overlap_threshold: fraction of a SAM2 segment's pixels that must
                overlap with the classical mask to be accepted. 10% is enough
                to accept crack-width expansions (Frangi center-line / full width
                ≈ 2 px / 8 px = 25%) while rejecting smooth-region blobs
                (face area / crack area << 1%).
            top_n_components: how many classical components (sorted by area,
                largest first) to use as SAM2 prompts.
        """
        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "sam2 is not installed.\n"
                "Install with:\n"
                "  pip install \"git+https://github.com/facebookresearch/sam2.git\"\n"
                "or:\n"
                "  pip install sam2"
            ) from exc

        import torch

        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.overlap_threshold = overlap_threshold
        self.top_n_components = top_n_components

        from sam2.sam2_image_predictor import SAM2ImagePredictor

        # `build_sam2` defaults device="cuda" — must pass it through from_pretrained
        # or the model is constructed on CUDA before we can move it, which asserts
        # on CPU-only torch builds.
        self._predictor = SAM2ImagePredictor.from_pretrained(model_id, device=self._device)
        try:
            self._predictor.model = self._predictor.model.to(self._device)
        except Exception:
            pass

    def expand(
        self,
        image: Image.Image,
        classical_mask: np.ndarray,
    ) -> np.ndarray:
        """Return a uint8 mask (0/255) that expands classical_mask via SAM2.

        Always includes at least the original classical_mask pixels. If SAM2
        adds nothing useful, the classical mask is returned unchanged.
        """
        import torch

        if classical_mask.max() == 0:
            return classical_mask.copy()

        rgb = np.array(image.convert("RGB"))

        classical_bool = classical_mask > 0

        n, _labels, stats, centroids = cv2.connectedComponentsWithStats(
            classical_mask, connectivity=8
        )
        if n <= 1:
            return classical_mask.copy()

        # Sort components by area descending, take top-N as SAM2 prompts.
        areas_and_centroids = [
            (
                int(stats[i, cv2.CC_STAT_AREA]),
                (float(centroids[i, 0]), float(centroids[i, 1])),
            )
            for i in range(1, n)
        ]
        areas_and_centroids.sort(reverse=True)
        top = areas_and_centroids[: self.top_n_components]

        expanded = classical_bool.copy()

        with torch.inference_mode():
            self._predictor.set_image(rgb)

            for _area, (cx, cy) in top:
                try:
                    masks, _scores, _logits = self._predictor.predict(
                        point_coords=np.array([[cx, cy]], dtype=float),
                        point_labels=np.array([1]),
                        multimask_output=True,
                    )
                except Exception:
                    continue

                # SAM2 returns 3 masks (small → large containment). Some SAM2
                # versions return float32 logit arrays rather than bool masks, so
                # normalize to bool first (logit > 0 is the standard decision
                # boundary; for 0/1 or bool arrays `> 0` is a no-op). Without this
                # the bitwise AND below raises TypeError on float32 and the whole
                # expansion is silently skipped. Prefer the smallest mask whose
                # overlap fraction meets the threshold — the tightest valid
                # expansion, avoiding creep into faces or smooth background.
                bool_masks = [np.asarray(m) > 0 for m in masks]
                order = np.argsort([int(m.sum()) for m in bool_masks])
                for idx in order:
                    m = bool_masks[idx]
                    area = int(m.sum())
                    if area == 0:
                        continue
                    overlap_frac = int((m & classical_bool).sum()) / area
                    if overlap_frac >= self.overlap_threshold:
                        expanded |= m
                        break

        return (expanded.astype(np.uint8)) * 255