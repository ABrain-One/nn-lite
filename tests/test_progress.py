"""Tests for the per-phone progress ledger."""
from ab.lite.progress import load_progress, progress_file, save_progress


def test_each_phone_gets_its_own_file(tmp_path):
    assert progress_file(tmp_path, "STK-L21").name == "processing_state_STK-L21.json"
    assert progress_file(tmp_path, "Redmi Note 9 Pro").name == "processing_state_Redmi_Note_9_Pro.json"
    assert progress_file(tmp_path, "SM-F926B") != progress_file(tmp_path, "STK-L21")


def test_missing_file_means_nothing_done_yet(tmp_path):
    assert load_progress(tmp_path / "none.json") == {"processed": [], "failed": []}


def test_round_trip(tmp_path):
    path = progress_file(tmp_path, "STK-L21")
    save_progress(path, {"processed": ["AirNet"], "failed": ["Broken"]})
    assert load_progress(path) == {"processed": ["AirNet"], "failed": ["Broken"]}


def test_old_shared_file_is_ignored(tmp_path):
    save_progress(tmp_path / "processing_state_dual.json", {"processed": ["AirNet"], "failed": []})
    assert load_progress(progress_file(tmp_path, "STK-L21"))["processed"] == []
