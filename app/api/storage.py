import base64
import io

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from PIL import Image

from app.core.job_state import get_job
from app.core.schemas import SaveImageRequest
from utils.minio import upload_to_minio

router = APIRouter()


@router.post("/save")
def save_image(body: SaveImageRequest):
    try:
        job = get_job(body.job_id)
        if not job:
            return JSONResponse(status_code=404, content={"message": "Job not found"})
        if job["status"] != "COMPLETED":
            return JSONResponse(status_code=400, content={"message": "Job is not complete yet"})

        # Strip data-URI prefix if present
        image_data = body.image
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        image_bytes = base64.b64decode(image_data)
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        object_name = upload_to_minio(pil_image, body.user_id)

        return {"message": "Image saved successfully", "object_name": object_name}

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})
