"""CodeFormer-based face restoration with adaptive fidelity weight and
feathered paste-back (via facelib FaceRestoreHelper)."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import normalize

# The vendored `codeformer` package uses absolute imports (`from codeformer.…`).
# Insert its parent on sys.path once so those resolve without rewriting 33 files.
_VENDORED_ROOT = Path(__file__).resolve().parent / "vendored"
if str(_VENDORED_ROOT) not in sys.path:
    sys.path.insert(0, str(_VENDORED_ROOT))

from codeformer.basicsr.archs import codeformer_arch  # noqa: F401  (registers arch)
from codeformer.basicsr.utils.registry import ARCH_REGISTRY
from codeformer.basicsr.utils.download_util import load_file_from_url
from codeformer.facelib.utils.face_restoration_helper import FaceRestoreHelper


_CODEFORMER_URL = (
    "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"
)
_WEIGHTS_DIR = Path(
    os.environ.get(
        "NV_FACE_WEIGHTS_DIR",
        str(Path(__file__).resolve().parent.parent.parent / "models"),
    )
)


def _cuda() -> bool:
    return torch.cuda.is_available()


def _img2tensor_bgr(face_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
    normalize(tensor, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
    return tensor.unsqueeze(0).to(device)


def _tensor2img_bgr(tensor: torch.Tensor) -> np.ndarray:
    t = tensor.squeeze(0).clamp(-1, 1).float().detach().cpu()
    t = (t + 1.0) * 0.5  # back to [0,1]
    arr = (t.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


class FaceRestorationAdapter:
    """CodeFormer + FaceRestoreHelper. Detects via RetinaFace, aligns to 512,
    restores at the chosen fidelity weight `w`, pastes back with feathered alpha."""

    def __init__(
        self,
        device: Optional[str] = None,
        face_size: int = 512,
        upscale_factor: int = 1,
    ) -> None:
        self.device = torch.device(device or ("cuda" if _cuda() else "cpu"))
        _WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = load_file_from_url(
            url=_CODEFORMER_URL,
            model_dir=str(_WEIGHTS_DIR),
            progress=True,
            file_name="codeformer.pth",
        )
        self.net = ARCH_REGISTRY.get("CodeFormer")(
            dim_embd=512,
            codebook_size=1024,
            n_head=8,
            n_layers=9,
            connect_list=["32", "64", "128", "256"],
        ).to(self.device)
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        state = ckpt.get("params_ema", ckpt)
        self.net.load_state_dict(state)
        self.net.eval()

        self.helper = FaceRestoreHelper(
            upscale_factor=upscale_factor,
            face_size=face_size,
            crop_ratio=(1, 1),
            det_model="retinaface_resnet50",
            save_ext="png",
            use_parse=True,
            device=self.device,
        )

    def detect_faces(
        self, image: Image.Image, only_center_face: bool = False
    ) -> int:
        bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        self.helper.clean_all()
        self.helper.read_image(bgr)
        num = self.helper.get_face_landmarks_5(
            only_center_face=only_center_face, resize=640, eye_dist_threshold=5
        )
        return int(num)

    def restore(
        self,
        image: Image.Image,
        fidelity_weight: float = 0.7,
        only_center_face: bool = False,
    ) -> tuple[Image.Image, int]:
        """Run detect → align → CodeFormer → feathered paste-back.

        Returns (restored_pil, num_faces). If no faces detected, returns the
        input unchanged and num_faces == 0.
        """
        bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        self.helper.clean_all()
        self.helper.read_image(bgr)
        num_faces = self.helper.get_face_landmarks_5(
            only_center_face=only_center_face, resize=640, eye_dist_threshold=5
        )
        if num_faces == 0:
            return image, 0

        self.helper.align_warp_face()

        for cropped_face in self.helper.cropped_faces:
            face_t = _img2tensor_bgr(cropped_face, self.device)
            try:
                with torch.no_grad():
                    output = self.net(
                        face_t, w=float(fidelity_weight), adain=True
                    )[0]
                restored = _tensor2img_bgr(output)
            except Exception:
                restored = cropped_face
            self.helper.add_restored_face(restored.astype("uint8"))

        # FaceRestoreHelper.paste_faces_to_input_image does inverse affine +
        # parsing-mask alpha + Gaussian feather at the seam internally.
        self.helper.get_inverse_affine(None)
        result_bgr = self.helper.paste_faces_to_input_image(
            upsample_img=None, draw_box=False
        )
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result_rgb), num_faces

    def detect_count(self, image: Image.Image) -> int:
        """Cheap call: just RetinaFace detect, no alignment or restoration."""
        bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        self.helper.clean_all()
        self.helper.read_image(bgr)
        return int(
            self.helper.get_face_landmarks_5(
                only_center_face=False, resize=640, eye_dist_threshold=5
            )
        )
