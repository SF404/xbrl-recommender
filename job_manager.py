# job_manager.py
from typing import Dict

jobs: Dict[str, dict] = {}  
# { job_id: {"status": "running/completed/failed", "progress": 0-100, "total": int, "done": int} }
