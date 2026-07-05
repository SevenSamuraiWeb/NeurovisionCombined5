import io
import time
from minio import Minio

from config import settings

minio_client = Minio(
    settings.minio_endpoint,
    access_key=settings.minio_access_key,
    secret_key=settings.minio_secret_key,
    secure=settings.minio_secure
)

BUCKET_NAME = settings.minio_bucket


def upload_to_minio(pil_image, user_id):

    buffer = io.BytesIO()

    pil_image.save(
        buffer,
        format="PNG"
    )

    buffer.seek(0)

    object_name = (
        f"users/{user_id}/enhanced/"
        f"enhanced_{int(time.time())}.png"
    )

    minio_client.put_object(
        bucket_name=BUCKET_NAME,
        object_name=object_name,
        data=buffer,
        length=buffer.getbuffer().nbytes,
        content_type="image/png"
    )

    return object_name