"""Latency measurements. Run twice per op — first pays lazy model load (cold),
second is warm. Results print in the session-end table (conftest).

On the OLD stack, blur/fog can't run through the jobs flow (dict-vs-PIL crash),
so they're timed directly against the Rajiv microservice (:8000) when it's up —
that's the baseline the merged in-process calls are compared to.
"""
import io
import os

import pytest
import requests

from conftest import BASE_URL, NEW_STACK, create_job, run_job_to_completion, timed

RAJIV_URL = os.environ.get("RAJIV_URL", "http://localhost:8000")

pytestmark = [pytest.mark.slow, pytest.mark.timeout(3600)]


def test_latency_health(http):
    for _ in range(5):
        with timed("GET / (health)"):
            http.get(f"{BASE_URL}/", timeout=10)


def test_latency_job_create(http, image_file):
    for _ in range(2):
        create_job(http, image_file, timer_name="POST /v1/analysis/jobs (classify)")


@pytest.mark.parametrize("stage", ["dark", "damaged"])
def test_latency_full_job(http, image_file, stage):
    for run in ("cold", "warm"):
        run_job_to_completion(http, image_file, stage,
                              timer_name=f"job execute+stream [{stage}] ({run})")


@pytest.mark.new_only
@pytest.mark.parametrize("stage", ["blurry", "foggy"])
def test_latency_full_job_new_stages(http, image_file, stage):
    for run in ("cold", "warm"):
        run_job_to_completion(http, image_file, stage,
                              timer_name=f"job execute+stream [{stage}] ({run})")


@pytest.mark.skipif(NEW_STACK, reason="old-stack baseline: direct microservice call")
@pytest.mark.parametrize("path,name", [("/predict", "blur"), ("/predict-fog", "fog")])
def test_latency_rajiv_direct(image_bytes, path, name):
    try:
        requests.get(f"{RAJIV_URL}/health", timeout=5)
    except requests.ConnectionError:
        pytest.skip(f"Rajiv microservice not running at {RAJIV_URL}")
    for run in ("cold", "warm"):
        with timed(f"microservice {name} direct ({run})"):
            r = requests.post(f"{RAJIV_URL}{path}",
                              files={"file": ("t.jpg", io.BytesIO(image_bytes), "image/jpeg")},
                              timeout=1800)
        assert r.status_code == 200, r.text


@pytest.mark.new_only
def test_latency_identifier(http, image_file):
    for run in ("cold", "warm"):
        with timed(f"POST /identifier ({run})"):
            r = http.post(f"{BASE_URL}/identifier", files=image_file(), timeout=600)
        assert r.status_code == 200


@pytest.mark.new_only
def test_latency_cloak(http, image_file):
    for run in ("cold", "warm"):
        with timed(f"POST /cloak fgsm ({run})"):
            r = http.post(f"{BASE_URL}/cloak", files=image_file(),
                          data={"method": "fgsm", "steps": "1"}, timeout=1800)
        assert r.status_code == 200