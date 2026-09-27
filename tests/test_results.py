"""Tests for benchmark_model output parsing and record construction.

These run without a phone, PyTorch or TensorFlow.
"""
import json

import pytest

from ab.lite.results import (
    benchmark_failed,
    build_record,
    extract_error_from_output,
    failed_result,
    parse_benchmark_output,
)

# Summary line printed by the LiteRT benchmark_model tool (values in microseconds).
OK_OUTPUT = """\
INFO: Initialized TensorFlow Lite runtime.
INFO: The input model file size (MB): 4.87
Running benchmark for at least 20 iterations and at least 1 seconds but terminate if exceeding 150 seconds.
count=20 first=31500 curr=30900 min=30718 max=36410 avg=31118.1 std=1323
Inference timings in us: Init: 21432, First inference: 35112, Warmup (avg): 31500, Inference (avg): 31118.1
"""

GPU_FAIL_OUTPUT = """\
INFO: Created TensorFlow Lite delegate for GPU.
ERROR: Following operations are not supported by GPU delegate:
ERROR: Failed to apply GPU delegate.
Benchmarking failed.
"""

MEMORY = {"total_ram_kb": 11631760, "free_ram_kb": 2818008, "available_ram_kb": 5356556, "cached_kb": 4672460}
ANALYTICS = {"timestamp": 0.0, "cpu_info": {"cpu_cores": 8}}


def ok(avg_us):
    return parse_benchmark_output(f"count=20 min={avg_us - 1} max={avg_us + 1} avg={avg_us} std=2")


# --- parsing -------------------------------------------------------------

def test_parse_converts_microseconds_to_nanoseconds():
    res = parse_benchmark_output(OK_OUTPUT)
    assert res["status"] == "ok"
    assert res["avg"] == pytest.approx(31118100.0)
    assert res["min"] == pytest.approx(30718000.0)
    assert res["max"] == pytest.approx(36410000.0)
    assert res["std"] == pytest.approx(1323000.0)


def test_benchmark_failed_detection():
    assert not benchmark_failed(OK_OUTPUT)
    assert benchmark_failed(GPU_FAIL_OUTPUT)
    assert benchmark_failed("")
    assert benchmark_failed("Failed to compute something\navg=1")


# --- error extraction ----------------------------------------------------

def test_extract_error_keeps_only_error_lines():
    msg = extract_error_from_output(GPU_FAIL_OUTPUT)
    assert msg == ("ERROR: Following operations are not supported by GPU delegate: | "
                   "ERROR: Failed to apply GPU delegate.")


def test_extract_error_keeps_at_most_three_lines():
    msg = extract_error_from_output("Error one\nCannot two\nFailed three\nUnsupported four")
    assert msg == "Error one | Cannot two | Failed three"


def test_extract_error_handles_empty_output():
    assert extract_error_from_output("") == "no output from benchmark_model"
    assert extract_error_from_output("   \n ") == "no output from benchmark_model"


def test_extract_error_falls_back_to_last_line():
    assert extract_error_from_output("starting\nsomething odd happened") == "something odd happened"


def test_extract_error_is_truncated():
    msg = extract_error_from_output("ERROR: " + "x" * 1000)
    assert len(msg) == 500
    assert msg.endswith("...")


# --- records -------------------------------------------------------------

def test_record_picks_fastest_backend():
    rec = build_record("AirNet", "SM-F926B", "14 | UP1A", 20,
                       {"cpu": ok(31000), "gpu": ok(8900), "npu": ok(9100)},
                       MEMORY, 128, ANALYTICS)
    assert rec["valid"] is True
    assert rec["emulator"] is False
    assert rec["unit"] == "GPU"
    assert rec["duration"] == 8900000
    assert rec["cpu_duration"] == 31000000
    assert (rec["in_dim_0"], rec["in_dim_1"], rec["in_dim_2"], rec["in_dim_3"]) == (1, 128, 128, 3)
    assert not any(k.endswith("_error") for k in rec)


def test_record_keeps_partial_failures():
    rec = build_record("AirNet", "Pixel", "13", 20,
                       {"cpu": ok(72000), "gpu": ok(10800), "npu": failed_result("NNAPI error")},
                       MEMORY, 32, ANALYTICS)
    assert rec["valid"] is True
    assert rec["unit"] == "GPU"
    assert rec["npu_duration"] == 0
    assert rec["npu_error"] == "NNAPI error"


def test_record_all_backends_failed_is_invalid_but_kept():
    rec = build_record("Broken", "Dev", "10", 20,
                       {b: failed_result(f"{b} failed") for b in ("cpu", "gpu", "npu")},
                       MEMORY, 32, ANALYTICS)
    assert rec["valid"] is False
    assert "duration" not in rec and "unit" not in rec
    assert [rec["cpu_error"], rec["gpu_error"], rec["npu_error"]] == ["cpu failed", "gpu failed", "npu failed"]
    assert rec["total_ram_kb"] == MEMORY["total_ram_kb"]


def test_record_field_order_is_stable():
    rec = build_record("AirNet", "Dev", "14", 20,
                       {"cpu": ok(3), "gpu": ok(2), "npu": failed_result("e")},
                       MEMORY, 64, ANALYTICS)
    assert list(rec) == [
        "model_name", "device_type", "os_version", "valid", "emulator", "iterations",
        "duration", "unit",
        "cpu_duration", "cpu_min_duration", "cpu_max_duration", "cpu_std_dev",
        "gpu_duration", "gpu_min_duration", "gpu_max_duration", "gpu_std_dev",
        "npu_duration", "npu_min_duration", "npu_max_duration", "npu_std_dev",
        "total_ram_kb", "free_ram_kb", "available_ram_kb", "cached_kb",
        "in_dim_0", "in_dim_1", "in_dim_2", "in_dim_3", "device_analytics",
        "npu_error",
    ]
    json.dumps(rec)  # must be serialisable
