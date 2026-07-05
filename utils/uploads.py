"""Shared multipart image-upload validation (previously copy-pasted in five
route files)."""
import io

from fastapi import HTTPException, UploadFile
from PIL import Image


def read_image_upload(file: UploadFile | None, *, key: str = "message") -> Image.Image:
    """Validate an uploaded image and return it as RGB PIL.

    Raises HTTPException(400) with the given envelope key ("message" for the
    jobs API, "error" for identifier/cloak) so error shapes stay per-API.
    """
    def bad(msg: str):
        return HTTPException(status_code=400, detail={key: msg})

    if file is None:
        raise bad("No image provided")
    if not file.filename:
        raise bad("Empty filename")
    if not (file.content_type or "").startswith("image/"):
        raise bad("File must be an image")

    try:
        return Image.open(io.BytesIO(file.file.read())).convert("RGB")
    except Exception as exc:
        raise bad(f"Failed to read image: {exc}")