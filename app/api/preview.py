from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse

router = APIRouter()

TEMP_DIR = Path("temp/jobs")


@router.get("/{job_id}/{stage}", name="get_preview")
def get_preview(job_id: str, stage: str):
    folder = TEMP_DIR / job_id

    candidates = (
        [folder / "final_preview.jpg"] if stage == "final"
        else [folder / f"stage_{stage}.jpg", folder / f"stage_{stage}.png"]
    )
    for path in candidates:
        if path.exists():
            mimetype = "image/jpeg" if path.suffix == ".jpg" else "image/png"
            return FileResponse(path.resolve(), media_type=mimetype)

    return JSONResponse(status_code=404, content={"message": "Preview not found"})
