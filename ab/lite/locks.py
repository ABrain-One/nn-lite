"""Locks that let several phones be benchmarked at the same time into one results folder.

Each phone is benchmarked by its own run of NN-Lite. The runs share the downloads in
``_work`` (the LEMUR database, the list of models, the CIFAR-10 images), which one run
at a time may download. Phones of the same model share a progress file and write the
same result files, so two of them cannot be benchmarked at the same time.

The locks are files next to what they protect. They are released when the run ends,
even if it crashes or is stopped.
"""
from contextlib import contextmanager
from pathlib import Path

from filelock import FileLock, Timeout

_claimed = []  # the locks taken by claim, kept until the run ends


@contextmanager
def downloading(work_dir):
    """Hold the lock on the shared downloads in ``work_dir``, waiting while another run holds it."""
    lock = FileLock(str(Path(work_dir) / "downloads.lock"))
    try:
        lock.acquire(timeout=0)
    except Timeout:
        print("[SETUP] Waiting for another run to finish downloading...")
        lock.acquire()
    try:
        yield
    finally:
        lock.release()


def claim(progress_path, device_model):
    """Lock the progress file of ``device_model`` until this run ends, and return the lock.

    A RuntimeError is raised if another run holds it.
    """
    lock = FileLock(str(progress_path) + ".lock")
    try:
        lock.acquire(timeout=0)
    except Timeout:
        raise RuntimeError(
            f"another run is already benchmarking a phone of model {device_model} into this results folder. "
            "Phones of the same model share one progress file and one result file per model, "
            "so benchmark them one after the other") from None
    _claimed.append(lock)
    return lock
