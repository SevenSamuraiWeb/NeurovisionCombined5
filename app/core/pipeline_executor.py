import threading
from pathlib import Path
from PIL import Image

from app.core.job_state import get_job, update_job
from app.core.events import publish
from app.blur.blur_service import de_blur
from app.dark.dark_service import de_dark
from app.fog.fog_service import de_fog
from app.old.old_service import de_old

# Canonical stage -> service mapping. Every service takes (PIL image, report=None)
# and returns a PIL image; report(substage, pil) streams intermediates.
STAGE_SERVICES = {
    "dark": de_dark,
    "blurry": de_blur,
    "foggy": de_fog,
    "damaged": de_old,
}

STAGE_ORDER = ["dark", "blurry", "foggy", "damaged"]

TEMP_DIR = Path("temp/jobs")


def _save_preview(job_id: str, stage: str, image: Image.Image) -> str:
    job_dir = TEMP_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    path = job_dir / f"stage_{stage}.jpg"
    image.save(str(path), format="JPEG", quality=85)
    return str(path)


def _finalize(job_id: str, image: Image.Image, message: str):
    job_dir = TEMP_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    final_path = str(job_dir / "final_preview.jpg")
    image.save(final_path, format="JPEG", quality=90)

    update_job(job_id, status="COMPLETED", progress=100, active_stage=None, final_path=final_path)
    publish(job_id, {
        "event": "restoration_completed",
        "job_id": job_id,
        "progress_percent": 100,
        "preview_url": f"/v1/preview/{job_id}/final",
        "message": message,
    })
    publish(job_id, {
        "event": "save_ready",
        "job_id": job_id,
        "message": "Image ready to save",
    })


def _run_pipeline(job_id: str):
    try:
        job = get_job(job_id)
        if not job:
            return

        current_image = Image.open(job["original_path"]).convert("RGB")

        pipeline = job["pipeline"] or []
        enabled_stages = [s for s in pipeline if s.get("enabled", True)]
        total = len(enabled_stages)

        update_job(job_id, status="PROCESSING", progress=0)
        publish(job_id, {
            "event": "pipeline_started",
            "job_id": job_id,
            "total_stages": total,
        })

        if total == 0:
            _finalize(job_id, current_image, "No degradations detected — original image returned")
            return

        for i, stage_config in enumerate(enabled_stages):
            stage_name = stage_config["stage"]
            progress_before = int(i / total * 100)

            update_job(job_id, active_stage=stage_name, progress=progress_before)
            publish(job_id, {
                "event": "stage_started",
                "job_id": job_id,
                "stage": stage_name,
                "progress_percent": progress_before,
                "message": f"Processing {stage_name}...",
            })

            if stage_name in STAGE_SERVICES:
                def _sub_report(substage, img, _stage=stage_name):
                    name = f"{_stage}_{substage}"
                    _save_preview(job_id, name, img)
                    publish(job_id, {
                        "event": "preview_ready",
                        "job_id": job_id,
                        "stage": name,
                        "preview_url": f"/v1/preview/{job_id}/{name}",
                    })

                current_image = STAGE_SERVICES[stage_name](current_image, report=_sub_report)

            _save_preview(job_id, stage_name, current_image)

            progress_after = int((i + 1) / total * 100)
            publish(job_id, {
                "event": "stage_completed",
                "job_id": job_id,
                "stage": stage_name,
                "progress_percent": progress_after,
                "preview_url": f"/v1/preview/{job_id}/{stage_name}",
                "message": f"Stage {stage_name} complete",
            })
            publish(job_id, {
                "event": "preview_ready",
                "job_id": job_id,
                "stage": stage_name,
                "preview_url": f"/v1/preview/{job_id}/{stage_name}",
            })

        _finalize(job_id, current_image, "Restoration complete")

    except Exception as e:
        update_job(job_id, status="FAILED", error_message=str(e))
        publish(job_id, {
            "event": "job_failed",
            "job_id": job_id,
            "message": str(e),
        })


def execute_job(job_id: str):
    update_job(job_id, status="QUEUED")
    thread = threading.Thread(target=_run_pipeline, args=(job_id,), daemon=True)
    thread.start()