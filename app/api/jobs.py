"""Jobs API — the single user flow:
classify -> user confirms/overrides pipeline -> execute -> SSE progress.

Merges the former analysis/pipeline/stream Flask blueprints (one resource,
one router).
"""
import json
from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from app.classifier.classifier import get_classifier, predict_image
from app.core.events import get_events_from, publish, subscribe, unsubscribe
from app.core.job_state import create_job, get_job, update_job
from app.core.pipeline_executor import STAGE_ORDER, execute_job
from app.core.schemas import PipelineUpdateRequest
from utils.uploads import read_image_upload

router = APIRouter()

TEMP_DIR = Path("temp/jobs")

_EDITABLE_STATUSES = {"CLASSIFIED", "WAITING_FOR_USER_CONFIRMATION"}
_TERMINAL_STATUSES = {"COMPLETED", "FAILED", "CANCELLED"}


def _build_pipeline(routes: list) -> list:
    return [
        {"stage": r, "enabled": True, "params": {}}
        for r in STAGE_ORDER if r in routes
    ]


@router.post("/jobs", status_code=201)
def create_analysis_job(image: UploadFile | None = File(None), id: str | None = Form(None)):
    try:
        pil_image = read_image_upload(image, key="message")
        if id is None:
            return JSONResponse(status_code=400, content={"message": "User id missing"})

        job_id = create_job(id)

        job_dir = TEMP_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        original_path = str(job_dir / "original.jpg")
        pil_image.save(original_path, format="JPEG", quality=95)
        update_job(job_id, original_path=original_path)

        publish(job_id, {"event": "analysis_started", "job_id": job_id})

        prediction = predict_image(model=get_classifier(), image_bytes=pil_image)
        routes = prediction.get("route", [])
        classification = {"probs": prediction["probs"], "routes": routes}

        pipeline = _build_pipeline(routes)
        status = "WAITING_FOR_USER_CONFIRMATION" if pipeline else "CLASSIFIED"

        update_job(job_id, status=status, classification=classification, pipeline=pipeline)
        publish(job_id, {"event": "classification_completed", "job_id": job_id, "routes": routes})
        publish(job_id, {"event": "pipeline_ready", "job_id": job_id})

        return {
            "job_id": job_id,
            "status": status,
            "classification": classification,
            "pipeline": pipeline,
        }
    except Exception as e:
        if hasattr(e, "status_code"):  # HTTPException from read_image_upload
            raise
        return JSONResponse(status_code=500, content={"message": str(e)})


@router.get("/jobs/{job_id}")
def get_analysis_job(job_id: str):
    job = get_job(job_id)
    if not job:
        return JSONResponse(status_code=404, content={"message": "Job not found"})

    response = {
        "job_id": job["id"],
        "user_id": job["user_id"],
        "status": job["status"],
        "progress": job["progress"],
        "active_stage": job["active_stage"],
        "classification": job["classification"],
        "pipeline": job["pipeline"],
        "error_message": job["error_message"],
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
    }
    if job["status"] == "COMPLETED":
        response["preview_url"] = f"/v1/preview/{job_id}/final"

    return response


@router.put("/jobs/{job_id}/pipeline")
def update_pipeline(job_id: str, body: PipelineUpdateRequest):
    job = get_job(job_id)
    if not job:
        return JSONResponse(status_code=404, content={"message": "Job not found"})
    if job["status"] not in _EDITABLE_STATUSES:
        return JSONResponse(
            status_code=400,
            content={"message": f"Cannot update pipeline in status '{job['status']}'"},
        )

    pipeline = [stage.model_dump() for stage in body.pipeline]
    update_job(job_id, pipeline=pipeline, status="WAITING_FOR_USER_CONFIRMATION")
    return {"job_id": job_id, "pipeline": pipeline}


@router.post("/jobs/{job_id}/execute", status_code=202)
def execute_pipeline(job_id: str):
    job = get_job(job_id)
    if not job:
        return JSONResponse(status_code=404, content={"message": "Job not found"})
    if job["status"] not in _EDITABLE_STATUSES:
        return JSONResponse(
            status_code=400,
            content={"message": f"Cannot execute job in status '{job['status']}'"},
        )

    execute_job(job_id)
    return {"job_id": job_id, "status": "QUEUED"}


@router.get("/jobs/{job_id}/stream")
def stream_job(job_id: str):
    if not get_job(job_id):
        return JSONResponse(status_code=404, content={"message": "Job not found"})

    def generate():
        offset, notify = subscribe(job_id)
        try:
            while True:
                triggered = notify.wait(timeout=25)
                notify.clear()

                new_events = get_events_from(job_id, offset)
                for ev in new_events:
                    yield f"data: {json.dumps(ev)}\n\n"
                offset += len(new_events)

                current_job = get_job(job_id)
                if current_job and current_job["status"] in _TERMINAL_STATUSES:
                    yield f"data: {json.dumps({'event': 'stream_closed', 'job_id': job_id})}\n\n"
                    break

                if not triggered:
                    # keepalive comment so proxies don't close the connection
                    yield ": keepalive\n\n"
        finally:
            # On client disconnect Starlette closes the generator; cleanup may
            # lag up to 25s while blocked in notify.wait — acceptable.
            unsubscribe(job_id, notify)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
