"""Tests for the locks that let several phones be benchmarked into one results folder."""
import subprocess
import sys
import threading
import time

import pytest

pytest.importorskip("filelock")
from ab.lite.locks import claim, downloading  # noqa: E402


def test_a_phone_model_is_benchmarked_by_one_run_at_a_time(tmp_path):
    state = tmp_path / "processing_state_Pixel_7.json"
    held = claim(state, "Pixel 7")
    other_run = [sys.executable, "-c", "import sys; from ab.lite.locks import claim\n"
                 "try: claim(sys.argv[1], 'Pixel 7')\nexcept RuntimeError as e: sys.exit(str(e))", str(state)]
    res = subprocess.run(other_run, capture_output=True, text=True)
    assert res.returncode == 1 and "benchmarking a phone of model Pixel 7" in res.stderr
    claim(tmp_path / "processing_state_SM-G991B.json", "SM-G991B").release()  # another model: free
    held.release()
    assert subprocess.run(other_run).returncode == 0


def test_the_lock_is_released_when_a_run_ends(tmp_path):
    state = tmp_path / "processing_state_Pixel_7.json"
    run = [sys.executable, "-c", "import sys; from ab.lite.locks import claim; claim(sys.argv[1], 'Pixel 7')",
           str(state)]
    assert subprocess.run(run).returncode == 0
    claim(state, "Pixel 7").release()


def test_downloads_wait_for_the_other_run(tmp_path, capsys):
    order = []

    def other_run():
        with downloading(tmp_path):
            order.append("other run starts")
            started.set()
            time.sleep(0.5)
            order.append("other run ends")

    started = threading.Event()
    thread = threading.Thread(target=other_run)
    thread.start()
    started.wait()
    with downloading(tmp_path):
        order.append("this run")
    thread.join()
    assert order == ["other run starts", "other run ends", "this run"]
    assert "Waiting for another run" in capsys.readouterr().out
