import sqlite3
import json
import uuid
from datetime import datetime
from pathlib import Path

DB_PATH = Path("neurovision.db")


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'CREATED',
            classification TEXT,
            pipeline TEXT,
            progress INTEGER DEFAULT 0,
            active_stage TEXT,
            error_message TEXT,
            original_path TEXT,
            final_path TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def create_job(user_id: str) -> str:
    job_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO jobs (id, user_id, status, created_at, updated_at) VALUES (?, ?, 'CREATED', ?, ?)",
        (job_id, user_id, now, now),
    )
    conn.commit()
    conn.close()
    return job_id


def get_job(job_id: str) -> dict | None:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    conn.close()
    if row is None:
        return None
    job = dict(row)
    if job.get("classification"):
        job["classification"] = json.loads(job["classification"])
    if job.get("pipeline"):
        job["pipeline"] = json.loads(job["pipeline"])
    return job


def update_job(job_id: str, **fields):
    if not fields:
        return
    fields["updated_at"] = datetime.utcnow().isoformat()
    for key in ("classification", "pipeline"):
        if key in fields and not isinstance(fields[key], str):
            fields[key] = json.dumps(fields[key])
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [job_id]
    conn = sqlite3.connect(DB_PATH)
    conn.execute(f"UPDATE jobs SET {set_clause} WHERE id = ?", values)
    conn.commit()
    conn.close()
