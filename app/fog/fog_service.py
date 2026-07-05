"""In-process fog/haze removal — replaces the HTTP microservice call."""
from functools import lru_cache

from PIL import Image

from app.fog.fog import FogRemovalPipeline


@lru_cache(maxsize=1)
def _pipeline() -> FogRemovalPipeline:
    # Lazy: constructing FogRemovalPipeline loads BSRGAN.
    return FogRemovalPipeline()


def de_fog(image: Image.Image, report=None) -> Image.Image:
    """Fog removal. report(substage_name, pil_image) receives intermediates."""
    dehazed, sr, final = _pipeline().remove_fog(image)
    if report:
        report("dehazed", dehazed)
        report("sr", sr)
    return final


def loaded() -> bool:
    return _pipeline.cache_info().currsize > 0