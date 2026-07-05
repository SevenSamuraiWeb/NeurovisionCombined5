"""Contract + latency tests. Framework-agnostic: run against any live backend.

Usage:
    # old stack (Flask :5000 + Rajiv :8000 + Nihaal :8001) — baseline
    pytest tests/ -m "not new_only" -s

    # new stack (FastAPI :5000)
    NEW_STACK=1 pytest tests/ -s

    BASE_URL   backend under test (default http://localhost:5000)
    NEW_STACK  set to 1 when testing the merged FastAPI backend
    -m "not minio"  to skip tests needing a running MinIO
    -m "not slow"   to skip full pipeline runs
"""
import io
import json
import os
import statistics
import time
from contextlib import contextmanager

import pytest
import requests
from PIL import Image

BASE_URL = os.environ.get("BASE_URL", "http://localhost:5000").rstrip("/")
NEW_STACK = os.environ.get("NEW_STACK", "") == "1"

_LATENCIES: list[tuple[str, float]] = []


def pytest_configure(config):
    config.addinivalue_line("markers", "new_only: endpoint/behavior only exists on the merged FastAPI backend")
    config.addinivalue_line("markers", "minio: requires a running MinIO at localhost:9000")
    config.addinivalue_line("markers", "slow: full model pipeline run")


def pytest_collection_modifyitems(config, items):
    if not NEW_STACK:
        skip = pytest.mark.skip(reason="new-stack only (set NEW_STACK=1)")
        for item in items:
            if "new_only" in item.keywords:
                item.add_marker(skip)


@pytest.fixture(scope="session")
def http():
    with requests.Session() as s:
        yield s


@pytest.fixture(scope="session")
def image_bytes():
    """Small synthetic RGB image — fast on CPU, deterministic."""
    img = Image.new("RGB", (256, 256))
    px = img.load()
    for y in range(256):
        for x in range(256):
            px[x, y] = (x, y, (x + y) // 2)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


@pytest.fixture(scope="session")
def image_file(image_bytes):
    def make():
        return {"image": ("test.jpg", io.BytesIO(image_bytes), "image/jpeg")}
    return make


@contextmanager
def timed(name):
    t0 = time.perf_counter()
    yield
    _LATENCIES.append((name, time.perf_counter() - t0))


@pytest.fixture
def timer():
    return timed


def create_job(http, image_file, timer_name=None):
    ctx = timed(timer_name) if timer_name else _null()
    with ctx:
        r = http.post(f"{BASE_URL}/v1/analysis/jobs",
                      files=image_file(), data={"id": "test-user"}, timeout=300)
    assert r.status_code == 201, r.text
    return r.json()


def run_job_to_completion(http, image_file, stage, timer_name=None):
    """create -> override pipeline to a single stage -> execute -> consume SSE."""
    job = create_job(http, image_file)
    job_id = job["job_id"]
    r = http.put(f"{BASE_URL}/v1/analysis/jobs/{job_id}/pipeline",
                 json={"pipeline": [{"stage": stage, "enabled": True, "params": {}}]},
                 timeout=30)
    assert r.status_code == 200, r.text

    ctx = timed(timer_name) if timer_name else _null()
    with ctx:
        r = http.post(f"{BASE_URL}/v1/analysis/jobs/{job_id}/execute", timeout=30)
        assert r.status_code == 202, r.text
        events = consume_stream(http, job_id)
    return job_id, events


def consume_stream(http, job_id, max_seconds=1800):
    """Read SSE until stream_closed; returns the list of event dicts."""
    events = []
    deadline = time.monotonic() + max_seconds
    with http.get(f"{BASE_URL}/v1/analysis/jobs/{job_id}/stream",
                  stream=True, timeout=(10, 60)) as r:
        assert r.status_code == 200, r.text
        for line in r.iter_lines(decode_unicode=True):
            if time.monotonic() > deadline:
                pytest.fail(f"SSE stream for {job_id} did not close within {max_seconds}s")
            if not line or not line.startswith("data:"):
                continue  # keepalive comments / blank separators
            ev = json.loads(line[len("data:"):].strip())
            events.append(ev)
            if ev.get("event") == "stream_closed":
                break
    return events


@contextmanager
def _null():
    yield


def pytest_sessionfinish(session, exitstatus):
    if not _LATENCIES:
        return
    by_name: dict[str, list[float]] = {}
    for name, secs in _LATENCIES:
        by_name.setdefault(name, []).append(secs)
    stack = "NEW (FastAPI)" if NEW_STACK else "OLD (Flask + microservices)"
    print(f"\n\n=== Latency report — {stack} @ {BASE_URL} ===")
    print(f"{'operation':<40} {'n':>3} {'median s':>10} {'min s':>10} {'max s':>10}")
    for name, vals in by_name.items():
        print(f"{name:<40} {len(vals):>3} {statistics.median(vals):>10.3f} "
              f"{min(vals):>10.3f} {max(vals):>10.3f}")
    print()