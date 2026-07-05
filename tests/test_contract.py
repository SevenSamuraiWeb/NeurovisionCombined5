"""API contract tests — response shapes and status codes the frontend depends on."""
import base64
import io

import pytest
import requests

from conftest import BASE_URL, create_job, run_job_to_completion


# --- health -----------------------------------------------------------------

def test_health(http):
    r = http.get(f"{BASE_URL}/", timeout=10)
    assert r.status_code == 200
    assert "message" in r.json()


# --- job creation validation ------------------------------------------------

def test_job_missing_image_400(http):
    r = http.post(f"{BASE_URL}/v1/analysis/jobs", data={"id": "u"}, timeout=10)
    assert r.status_code == 400
    assert "message" in r.json()


def test_job_missing_id_400(http, image_file):
    r = http.post(f"{BASE_URL}/v1/analysis/jobs", files=image_file(), timeout=10)
    assert r.status_code == 400
    assert "message" in r.json()


def test_job_bad_mimetype_400(http):
    r = http.post(f"{BASE_URL}/v1/analysis/jobs",
                  files={"image": ("x.txt", io.BytesIO(b"not an image"), "text/plain")},
                  data={"id": "u"}, timeout=10)
    assert r.status_code == 400


# --- job lifecycle ----------------------------------------------------------

def test_create_and_get_job(http, image_file):
    job = create_job(http, image_file)
    assert job["job_id"]
    assert job["status"] in {"CLASSIFIED", "WAITING_FOR_USER_CONFIRMATION"}
    assert set(job["classification"]) == {"probs", "routes"}
    assert isinstance(job["pipeline"], list)

    r = http.get(f"{BASE_URL}/v1/analysis/jobs/{job['job_id']}", timeout=10)
    assert r.status_code == 200
    body = r.json()
    for key in ("job_id", "user_id", "status", "progress", "active_stage",
                "classification", "pipeline", "error_message", "created_at", "updated_at"):
        assert key in body, f"missing {key}"


def test_get_unknown_job_404(http):
    r = http.get(f"{BASE_URL}/v1/analysis/jobs/doesnotexist", timeout=10)
    assert r.status_code == 404


def test_pipeline_override(http, image_file):
    job = create_job(http, image_file)
    job_id = job["job_id"]
    body = {"pipeline": [{"stage": "dark", "enabled": True, "params": {}}]}
    r = http.put(f"{BASE_URL}/v1/analysis/jobs/{job_id}/pipeline", json=body, timeout=10)
    assert r.status_code == 200
    assert r.json()["pipeline"][0]["stage"] == "dark"

    # override round-trips through GET (enabled/params preserved)
    got = http.get(f"{BASE_URL}/v1/analysis/jobs/{job_id}", timeout=10).json()
    assert got["pipeline"] == body["pipeline"]

    r = http.put(f"{BASE_URL}/v1/analysis/jobs/{job_id}/pipeline", json={"nope": 1}, timeout=10)
    assert r.status_code == 400
    assert "message" in r.json()


def test_pipeline_override_unknown_job_404(http):
    r = http.put(f"{BASE_URL}/v1/analysis/jobs/doesnotexist/pipeline",
                 json={"pipeline": []}, timeout=10)
    assert r.status_code == 404


def test_stream_unknown_job_404(http):
    r = http.get(f"{BASE_URL}/v1/analysis/jobs/doesnotexist/stream", timeout=10)
    assert r.status_code == 404


def test_preview_404(http):
    r = http.get(f"{BASE_URL}/v1/preview/doesnotexist/final", timeout=10)
    assert r.status_code == 404


# --- full pipeline runs (SSE) -----------------------------------------------

def _assert_job_events(http, job_id, events, stage):
    names = [e.get("event") for e in events]
    assert "restoration_completed" in names, f"job did not complete: {names}"
    assert "stream_closed" in names
    assert any(e.get("event") == "stage_completed" and e.get("stage") == stage for e in events)
    # every preview_url must serve an image
    for ev in events:
        url = ev.get("preview_url")
        if url:
            r = http.get(f"{BASE_URL}{url}" if url.startswith("/") else url, timeout=30)
            assert r.status_code == 200, f"preview {url} -> {r.status_code}"
            assert r.headers["Content-Type"].startswith("image/")


@pytest.mark.slow
@pytest.mark.timeout(1800)
def test_full_job_dark(http, image_file):
    job_id, events = run_job_to_completion(http, image_file, "dark")
    _assert_job_events(http, job_id, events, "dark")


@pytest.mark.slow
@pytest.mark.new_only  # crashes on the old stack: de_blur returned a dict, executor expected PIL
@pytest.mark.timeout(1800)
def test_full_job_blurry(http, image_file):
    job_id, events = run_job_to_completion(http, image_file, "blurry")
    _assert_job_events(http, job_id, events, "blurry")


@pytest.mark.slow
@pytest.mark.new_only  # same dict-vs-PIL crash on the old stack
@pytest.mark.timeout(1800)
def test_full_job_foggy(http, image_file):
    job_id, events = run_job_to_completion(http, image_file, "foggy")
    _assert_job_events(http, job_id, events, "foggy")


@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_full_job_damaged(http, image_file):
    job_id, events = run_job_to_completion(http, image_file, "damaged")
    _assert_job_events(http, job_id, events, "damaged")


@pytest.mark.slow
@pytest.mark.new_only
@pytest.mark.timeout(3600)
def test_damaged_substage_previews(http, image_file):
    """Merged backend streams old-photo sub-stage previews as preview_ready events."""
    _, events = run_job_to_completion(http, image_file, "damaged")
    substages = [e["stage"] for e in events
                 if e.get("event") == "preview_ready" and e.get("stage", "").startswith("damaged_")]
    assert substages, "expected damaged_<substage> preview_ready events"


# --- storage ----------------------------------------------------------------

def test_storage_save_validation_400(http):
    r = http.post(f"{BASE_URL}/v1/storage/save", json={}, timeout=10)
    assert r.status_code == 400
    assert "message" in r.json()


@pytest.mark.minio
@pytest.mark.slow
@pytest.mark.timeout(1800)
def test_storage_save(http, image_file, image_bytes):
    job_id, _ = run_job_to_completion(http, image_file, "dark")
    data_url = "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode()
    r = http.post(f"{BASE_URL}/v1/storage/save",
                  json={"job_id": job_id, "user_id": "test-user", "image": data_url},
                  timeout=60)
    assert r.status_code == 200, r.text
    assert "object_name" in r.json()


# --- identifier + cloak (merged backend only) --------------------------------

@pytest.mark.new_only
def test_identifier(http, image_file):
    r = http.post(f"{BASE_URL}/identifier", files=image_file(), timeout=300)
    assert r.status_code == 200, r.text
    body = r.json()
    for key in ("top_label", "top_label_short", "confidence", "all_scores"):
        assert key in body
    assert body["all_scores"] and {"label", "short", "score"} <= set(body["all_scores"][0])

    r = http.post(f"{BASE_URL}/identifier", timeout=10)
    assert r.status_code == 400
    assert "error" in r.json()

    assert http.get(f"{BASE_URL}/identifier/health", timeout=10).status_code == 200


@pytest.mark.new_only
@pytest.mark.slow
@pytest.mark.timeout(1800)
def test_cloak(http, image_file):
    r = http.post(f"{BASE_URL}/cloak", files=image_file(),
                  data={"method": "fgsm", "steps": "1"}, timeout=1200)
    assert r.status_code == 200, r.text
    body = r.json()
    for key in ("cloaked_image_base64", "metrics", "faces_found", "protection_mode"):
        assert key in body
    assert {"epsilon_used", "method", "steps"} <= set(body["metrics"])

    r = http.post(f"{BASE_URL}/cloak", files=image_file(), data={"method": "bogus"}, timeout=60)
    assert r.status_code == 400
    assert "error" in r.json()

    assert http.get(f"{BASE_URL}/cloak/health", timeout=30).status_code == 200