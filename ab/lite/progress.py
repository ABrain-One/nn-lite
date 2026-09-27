"""Per-phone progress ledger that makes benchmarking runs resumable.

Each phone gets its own file (``processing_state_<device>.json``) listing the
models already processed or failed on it, so benchmarking a second phone does
not skip models that were only measured on the first one.
"""
import json
import re
from pathlib import Path


def progress_file(work_dir, device):
    """Path of the progress file for ``device`` inside ``work_dir``."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", device.strip()) or "unknown_device"
    return Path(work_dir) / f"processing_state_{safe}.json"


def load_progress(path):
    """Load a progress file, or return an empty ledger if it does not exist."""
    path = Path(path)
    state = json.loads(path.read_text()) if path.exists() else {}
    state.setdefault("processed", [])
    state.setdefault("failed", [])
    return state


def save_progress(path, state):
    Path(path).write_text(json.dumps(state, indent=2))
