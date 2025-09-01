from threading import Lock

jobs = {}
_jobs_lock = Lock()

def job_set(job_id: str, payload: dict):
    with _jobs_lock:
        jobs[job_id] = payload

def job_get(job_id: str, default=None):
    with _jobs_lock:
        return jobs.get(job_id, default)

def job_update(job_id: str, **kwargs):
    with _jobs_lock:
        if job_id in jobs:
            jobs[job_id].update(kwargs)
