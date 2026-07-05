"""In-process blur/scratch restoration — replaces the HTTP microservice call."""
from functools import lru_cache

import numpy as np
from PIL import Image

from app.blur.deblur import RestorePipeline


@lru_cache(maxsize=1)
def _pipeline() -> RestorePipeline:
    # Lazy: constructing RestorePipeline loads U-Net + LaMa + NAFNet + BSRGAN
    # (and downloads BSRGAN if missing) — don't pay that at import time.
    return RestorePipeline()


def de_blur(image: Image.Image, report=None) -> Image.Image:
    """Blur removal. report(substage_name, pil_image) receives intermediates."""
    trained, pretrained, final, damage_map = _pipeline().restore(image)
    if report:
        report("trained", trained)
        report("pretrained", pretrained)
        if damage_map is not None:
            report("mask", Image.fromarray((damage_map * 255).astype(np.uint8)))
    return final


def loaded() -> bool:
    return _pipeline.cache_info().currsize > 0
