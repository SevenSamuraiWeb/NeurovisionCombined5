from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from simple_lama_inpainting import SimpleLama

from app.old.colorizer import ColorizationAdapter, needs_colorize
from app.old.common import DEFAULT_MODEL_PATH, DEFAULT_RUNS_DIR, adaptive_tile_size, is_grayscale
from app.old.damage_detector import detect_damage_mask
from app.old.diffusion_fallback import (
    DiffusionSRAdapter,
    diffusion_enabled,
    make_adapter_if_enabled,
    should_use_diffusion,
)
from app.old.face_restorer import FaceRestorationAdapter
from app.old.modeling import PredictionResult, load_model, predict_image
from app.old.preclean import PreCleanAdapter, needs_preclean
from app.old.sam_expander import SamMaskExpander, sam_available
from app.old.severity import codeformer_fidelity_weight, compute_severity

_REALESRGAN_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
)


def _cuda_available() -> bool:
    return torch.cuda.is_available()


@dataclass
class PipelineArtifacts:
    run_id: str
    output_dir: Path
    final_image_path: Path
    metadata_path: Path
    stages: Dict[str, str]
    predictions: Dict[str, object]
    metadata_dict: Dict[str, object] = field(default_factory=dict)
    # Stage images kept in memory so callers can stream/serve them without
    # re-reading from disk.
    memory_stages: Dict[str, Image.Image] = field(default_factory=dict)


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def bgr_to_pil(image: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def _strip_chroma_from_face(restored: Image.Image, pre_codeformer: Image.Image) -> Image.Image:
    """Replace restored image's a/b (chroma) channels with those of pre_codeformer.

    CodeFormer's prior is RGB-trained and leaks hallucinated chrominance (bluish
    lips, tinted eyes) onto monochrome inputs. When the source is grayscale we
    keep CodeFormer's L (restored detail) but discard its a/b. Operating in LAB
    is enough — no need to mask out the face region, since `pre_codeformer` is
    the same monochrome image everywhere else, so the a/b channels match
    outside the face by construction.
    """
    if restored.size != pre_codeformer.size:
        pre_codeformer = pre_codeformer.resize(restored.size, Image.BICUBIC)
    r_bgr = cv2.cvtColor(np.asarray(restored.convert("RGB")), cv2.COLOR_RGB2BGR)
    p_bgr = cv2.cvtColor(np.asarray(pre_codeformer.convert("RGB")), cv2.COLOR_RGB2BGR)
    r_lab = cv2.cvtColor(r_bgr, cv2.COLOR_BGR2LAB)
    p_lab = cv2.cvtColor(p_bgr, cv2.COLOR_BGR2LAB)
    r_lab[..., 1:] = p_lab[..., 1:]
    out_bgr = cv2.cvtColor(r_lab, cv2.COLOR_LAB2BGR)
    return Image.fromarray(cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB))


class RealESRGANAdapter:
    def __init__(self, scale: int = 4, tile: int | None = None) -> None:
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=23, num_grow_ch=32, scale=scale,
        )
        use_gpu = _cuda_available()
        if tile is None:
            tile = adaptive_tile_size()
        self.upsampler = RealESRGANer(
            scale=scale,
            model_path=_REALESRGAN_URL,
            model=model,
            tile=tile,
            tile_pad=10,
            pre_pad=0,
            half=use_gpu,
            device="cuda" if use_gpu else "cpu",
        )

    def enhance(self, image: Image.Image, max_long_edge: int = 2048) -> Image.Image:
        # Pre-downscale so the 4× output lands at max_long_edge rather than
        # overshooting and immediately discarding resolution. For an 860 px
        # input this cuts tile count from 6 → 2 (~3× speedup on CPU) with no
        # meaningful quality loss on degraded old photos.
        target_in = max_long_edge // self.upsampler.scale
        in_long = max(image.size)
        if in_long > target_in:
            pre_scale = target_in / in_long
            image = image.resize(
                (round(image.width * pre_scale), round(image.height * pre_scale)),
                Image.Resampling.LANCZOS,
            )
        bgr = pil_to_bgr(image)
        output, _ = self.upsampler.enhance(bgr, outscale=self.upsampler.scale)
        result = bgr_to_pil(output)
        long_edge = max(result.size)
        if long_edge > max_long_edge:
            scale = max_long_edge / long_edge
            new_size = (round(result.width * scale), round(result.height * scale))
            result = result.resize(new_size, Image.Resampling.LANCZOS)
        return result


class LaMaInpaintingAdapter:
    def __init__(self) -> None:
        self.model = SimpleLama()

    def inpaint(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        if mask.mean() == 0:
            return image
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        if mask.shape[:2] != (image.height, image.width):
            mask = np.array(
                Image.fromarray(mask).resize(image.size, Image.NEAREST)
            )
        # Dilate 4 px so LaMa has enough edge context to avoid seams at mask boundaries.
        kernel = np.ones((9, 9), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask_img = Image.fromarray(mask).convert("L")
        return self.model(image.convert("RGB"), mask_img)


class OldPhotoRestorationPipeline:
    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        runs_dir: str | Path = DEFAULT_RUNS_DIR,
        device: Optional[str] = None,
    ) -> None:
        self.model = load_model(model_path, device=device)
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.global_enhancer = RealESRGANAdapter()
        self.inpainter = LaMaInpaintingAdapter()
        self.face_enhancer = FaceRestorationAdapter(device=device)
        self._preclean: Optional[PreCleanAdapter] = None
        self._diffusion: Optional[DiffusionSRAdapter] = None
        self._diffusion_init_attempted = False
        self._colorizer: Optional[ColorizationAdapter] = None
        self._sam: Optional[SamMaskExpander] = None
        self._sam_init_attempted = False
        self._device = device

    def _get_preclean(self) -> PreCleanAdapter:
        if self._preclean is None:
            self._preclean = PreCleanAdapter(device=self._device)
        return self._preclean

    def _get_diffusion(self) -> Optional[DiffusionSRAdapter]:
        if self._diffusion is not None:
            return self._diffusion
        if self._diffusion_init_attempted:
            return None
        self._diffusion_init_attempted = True
        self._diffusion = make_adapter_if_enabled()
        return self._diffusion

    def _get_sam(self) -> Optional[SamMaskExpander]:
        if self._sam is not None:
            return self._sam
        if self._sam_init_attempted:
            return None
        self._sam_init_attempted = True
        if not sam_available():
            print("[pipeline] segment-anything not installed — SAM expansion disabled.", flush=True)
            return None
        try:
            self._sam = SamMaskExpander(device=self._device)
            print("[pipeline] SAM2 mask expander loaded (hiera-base-plus).", flush=True)
        except Exception as exc:
            print(f"[pipeline] SAM init failed, skipping: {exc}", flush=True)
        return self._sam

    def _get_colorizer(self) -> ColorizationAdapter:
        if self._colorizer is None:
            self._colorizer = ColorizationAdapter(device=self._device)
        return self._colorizer

    def _prediction_to_dict(self, prediction: PredictionResult) -> Dict[str, object]:
        return {
            "predicted_types": prediction.predicted_types,
            "type_scores": prediction.type_scores,
            "damage_ratio": prediction.damage_ratio,
            "recommended_steps": prediction.recommended_steps,
        }

    def predict(self, image: Image.Image) -> PredictionResult:
        num_faces = self.face_enhancer.detect_count(image)
        return predict_image(self.model, image, face_detected=num_faces > 0)

    def run(
        self,
        image: Image.Image,
        on_progress: Optional[Callable[[str, Image.Image], None]] = None,
    ) -> PipelineArtifacts:
        run_id = uuid.uuid4().hex[:12]
        output_dir = self.runs_dir / run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        prediction = self.predict(image)
        stages: Dict[str, str] = {}
        memory_stages: Dict[str, Image.Image] = {}
        current = image.convert("RGB")

        def _report(stage_name: str, img: Image.Image, filename: str):
            path = output_dir / filename
            img.save(path)
            stages[stage_name] = str(path)
            memory_stages[stage_name] = img.copy()
            if on_progress:
                on_progress(stage_name, img)

        _report("input", current, "input.png")
        _report("predicted_mask", Image.fromarray(prediction.damage_mask), "predicted_mask.png")

        image_clean = not prediction.predicted_types
        lama_skipped_runaway_mask = False

        # [1] Damage detection on the ORIGINAL input.
        # Critical ordering: must run BEFORE pre-clean. SCUNet smooths
        # high-frequency content including scratches/creases, so Frangi
        # vesselness sees a clean image and fails to detect anything.
        # Detection ALWAYS runs (regardless of the classifier's predicted_types):
        # the classifier frequently omits 'scratches'/'missing_patch' on real OOD
        # photos even when physical damage is obvious, so gating on image_clean
        # would silently skip inpainting on visibly damaged inputs. The detector
        # self-gates only the stain pass on predicted_types and prunes any
        # component over 25% coverage, bounding over-detection on clean images.
        damage_detection = detect_damage_mask(
            current,
            model_mask=prediction.damage_mask,
            model_damage_ratio=prediction.damage_ratio,
            predicted_types=prediction.predicted_types,
            # SAM expansion disabled: BOPBTL's mask is precise enough to feed
            # LaMa directly. SAM2's "segment object at this prompt" semantics
            # frequently snap to the whole background when prompted on long
            # connected scratch networks, producing runaway >50% coverage.
            sam_expander=None,
        )
        _report("robust_damage_mask", Image.fromarray(damage_detection.mask), "robust_damage_mask.png")

        # [2] Inpaint on the ORIGINAL — LaMa is most accurate on un-smoothed input.
        if damage_detection.coverage > 0.5:
            lama_skipped_runaway_mask = True
        elif damage_detection.coverage > 0.0:
            current = self.inpainter.inpaint(current, damage_detection.mask)
            _report("after_lama_inpainting", current, "after_lama_inpainting.png")

        # [3] Pre-clean (SCUNet) on the inpainted image.
        # Runs after inpaint so scratches are gone first; preclean then
        # removes residual noise/JPEG/grain without losing structured damage.
        # Severity depends only on the (immutable) prediction and damage
        # detection, so it's computed once and reused for both the pre-clean
        # decision and downstream SR/diffusion/CodeFormer choices.
        severity = compute_severity(prediction, damage_detection)
        preclean_used = needs_preclean(prediction.predicted_types, severity)
        if preclean_used:
            current = self._get_preclean().clean(current)
            _report("after_preclean", current, "after_preclean.png")

        sr_backend = "none"
        if "real_esrgan" in prediction.recommended_steps:
            adapter = self._get_diffusion() if should_use_diffusion(severity) else None
            if adapter is not None:
                current = adapter.enhance(current)
                _report("after_diffusion_sr", current, "after_diffusion_sr.png")
                sr_backend = "diffusion"
            else:
                current = self.global_enhancer.enhance(current)
                _report("after_real_esrgan", current, "after_real_esrgan.png")
                sr_backend = "real_esrgan"

        fidelity_w = codeformer_fidelity_weight(severity)
        # Capture the pre-CodeFormer image so we can restore chrominance after
        # restore on monochrome inputs (see _strip_chroma_from_face).
        input_is_grayscale = is_grayscale(image)
        pre_codeformer = current
        restored, num_faces = self.face_enhancer.restore(
            current, fidelity_weight=fidelity_w
        )
        if num_faces > 0:
            if input_is_grayscale:
                restored = _strip_chroma_from_face(restored, pre_codeformer)
            current = restored
            _report("after_face_restore", current, "after_face_restore.png")

        # Colorize B&W / sepia inputs. Detection is on the ORIGINAL image:
        # post-restoration the image may already have colors hallucinated by
        # CodeFormer's prior, but if the source was monochrome the user wants
        # full colorization, not partial.
        colorize_used = needs_colorize(image)
        if colorize_used:
            try:
                current = self._get_colorizer().colorize(current)
                _report("after_colorize", current, "after_colorize.png")
            except Exception as exc:
                print(f"[pipeline] colorize failed, skipping: {exc}", flush=True)
                colorize_used = False

        _report("final", current, "final.png")
        final_path = Path(stages["final"])

        metadata_path = output_dir / "metadata.json"
        metadata = {
            "run_id": run_id,
            "image_clean": image_clean,
            "lama_skipped_runaway_mask": lama_skipped_runaway_mask,
            "damage_detection": (
                {"source": damage_detection.source, "coverage": damage_detection.coverage}
                if damage_detection is not None
                else None
            ),
            "severity": severity,
            "preclean_used": preclean_used,
            "sr_backend": sr_backend,
            "diffusion_enabled_env": diffusion_enabled(),
            "codeformer_fidelity_weight": fidelity_w,
            "num_faces": num_faces,
            "colorize_used": colorize_used,
            "predictions": self._prediction_to_dict(prediction),
            "stages": stages,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        return PipelineArtifacts(
            run_id=run_id,
            output_dir=output_dir,
            final_image_path=final_path,
            metadata_path=metadata_path,
            stages=stages,
            predictions=metadata["predictions"],
            metadata_dict=metadata,
            memory_stages=memory_stages,
        )
