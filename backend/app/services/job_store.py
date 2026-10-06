from datetime import datetime, timedelta, timezone
from pathlib import Path
import threading
import uuid
import shutil

from app.config import settings


class JobRegistry:
    def __init__(self):
        self._lock = threading.RLock()
        self._jobs: dict[str, dict] = {}

    def create(self, kind: str, **values) -> str:
        with self._lock:
            self._prune()
            job_id = uuid.uuid4().hex
            self._jobs[job_id] = {"job_id": job_id, "kind": kind, "status": "queued", "progress": 0, "stage": "Queued", "logs": [], "created_at": datetime.now(timezone.utc), **values}
            return job_id

    def update(self, job_id: str, **values) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(values)

    def log(self, job_id: str, message: str, level: str = "info") -> None:
        with self._lock:
            if job_id not in self._jobs:
                return
            entries = self._jobs[job_id].setdefault("logs", [])
            entries.append({"timestamp": datetime.now(timezone.utc).isoformat(), "level": level, "message": message})
            del entries[:-100]

    def clear(self) -> None:
        with self._lock:
            self._jobs.clear()

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            self._prune()
            record = self._jobs.get(job_id)
            if not record:
                return None
            snapshot = dict(record)
            snapshot["logs"] = list(record.get("logs", []))
            return snapshot

    def _prune(self) -> None:
        expire_before = datetime.now(timezone.utc) - timedelta(hours=settings.job_retention_hours)
        expired = [job_id for job_id, record in self._jobs.items() if record["created_at"] < expire_before]
        for job_id in expired:
            record = self._jobs.pop(job_id)
            if record.get("work_dir"):
                shutil.rmtree(record["work_dir"], ignore_errors=True)
            if record.get("clip_path"):
                Path(record["clip_path"]).unlink(missing_ok=True)


jobs = JobRegistry()
