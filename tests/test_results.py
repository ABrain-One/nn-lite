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

# Output of the LiteRT benchmark_model tool (times in microseconds). It prints statistics
# for a warm-up phase and then for the timed runs; the values here differ on purpose.
WARMUP = ("INFO: Running benchmark for at least 1 iterations and at least 0.5 seconds "
          "but terminate if exceeding 150 seconds.\n")
TIMED = ("INFO: Running benchmark for at least 20 iterations and at least 0 seconds "
         "but terminate if exceeding 3600 seconds.\n")
WARMED_UP = WARMUP + "INFO: count=1 curr=40000\n\n" + TIMED  # prefix for the statistics of the timed runs
OK_OUTPUT = (
    "INFO: Initialized TensorFlow Lite runtime.\n"
    "INFO: The input model file size (MB): 4.87\n"
    + WARMUP +
    "INFO: count=9 first=91020 curr=55300 min=54900 max=91020 avg=60125.5 std=11800\n\n"
    + TIMED +
    "INFO: count=20 first=31500 curr=30900 min=30718 max=36410 avg=31118.1 std=1323\n\n"
    "INFO: Inference timings in us: Init: 21432, First inference: 91020, "
    "Warmup (avg): 60125.5, Inference (avg): 31118.1\n"
)

GPU_FAIL_OUTPUT = """\
INFO: Created TensorFlow Lite delegate for GPU.
ERROR: Following operations are not supported by GPU delegate:
ERROR: Failed to apply GPU delegate.
Benchmarking failed.
"""

MEMORY = {"total_ram_kb": 11631760, "free_ram_kb": 2818008, "available_ram_kb": 5356556, "cached_kb": 4672460}
ANALYTICS = {"timestamp": 0.0, "cpu_info": {"cpu_cores": 8}}


def ok(avg_us):
    return parse_benchmark_output(f"{WARMED_UP}count=20 first={avg_us} curr={avg_us} "
                                  f"min={avg_us - 1} max={avg_us + 1} avg={avg_us} std=2")


# --- parsing -------------------------------------------------------------

def test_parse_reads_the_timed_runs_not_the_warm_up():
    res = parse_benchmark_output(OK_OUTPUT)
    assert res["status"] == "ok"
    assert res["runs"] == 20
    assert res["avg"] == pytest.approx(31118100.0)  # the warm-up average is 60125.5 us
    assert res["min"] == pytest.approx(30718000.0)
    assert res["max"] == pytest.approx(36410000.0)
    assert res["std"] == pytest.approx(1323000.0)


# Real output of the bundled benchmark_model on an STK-L21 (CPU, a CIFAR-10 model taking about
# 2.2 s per inference), with NN-Lite's former default flags and with the flags it now uses.
REAL_STK_L21_DEFAULT = (
    "INFO: Running benchmark for at least 1 iterations and at least 0.5 seconds but terminate if exceeding 150 seconds.\n"
    "INFO: count=1 curr=2090305 p5=2090305 median=2090305 p95=2090305\n\n"
    "INFO: Running benchmark for at least 20 iterations and at least 1 seconds but terminate if exceeding 150 seconds.\n"
    "INFO: count=20 first=2204662 curr=2117115 min=2074572 max=2556951 avg=2.20029e+06 std=135651 "
    "p5=2074714 median=2142188 p95=2556951\n\n"
)
REAL_STK_L21_FIXED = (
    "INFO: Running benchmark for at least 1 iterations and at least 0.5 seconds but terminate if exceeding 3600 seconds.\n"
    "INFO: count=1 curr=2136623 p5=2136623 median=2136623 p95=2136623\n\n"
    "INFO: Running benchmark for at least 20 iterations and at least 0 seconds but terminate if exceeding 3600 seconds.\n"
    "INFO: count=20 first=2180398 curr=2261945 min=2157441 max=3097748 avg=2.34979e+06 std=235799 "
    "p5=2170335 median=2246033 p95=3097748\n\n"
)


@pytest.mark.parametrize("out, avg_us, min_us, max_us, std_us", [
    (REAL_STK_L21_DEFAULT, 2.20029e6, 2074572, 2556951, 135651),
    (REAL_STK_L21_FIXED, 2.34979e6, 2157441, 3097748, 235799),
])
def test_parse_real_output_of_the_bundled_binary(out, avg_us, min_us, max_us, std_us):
    assert not benchmark_failed(out)
    res = parse_benchmark_output(out)
    assert res["runs"] == 20
    assert res["avg"] == pytest.approx(avg_us * 1000)   # 2.2 s, formerly read as 2.2 microseconds
    assert (res["min"], res["max"], res["std"]) == (min_us * 1000, max_us * 1000, std_us * 1000)


# Real output for a very fast model (about 0.4 ms) on the same STK-L21, with the former default
# flags: the warm-up phase ran 1024 times and its statistics differ strongly from the 2487 timed runs.
REAL_STK_L21_FAST_DEFAULT = (
    "INFO: Running benchmark for at least 1 iterations and at least 0.5 seconds but terminate if exceeding 150 seconds.\n"
    "INFO: count=1024 first=8993 curr=457 min=385 max=8993 avg=483.029 std=330 p5=387 median=431 p95=1008\n\n"
    "INFO: Running benchmark for at least 20 iterations and at least 1 seconds but terminate if exceeding 150 seconds.\n"
    "INFO: count=2487 first=502 curr=388 min=385 max=585 avg=400.341 std=17 p5=387 median=394 p95=436\n\n"
)
REAL_STK_L21_FAST_FIXED = (
    "INFO: Running benchmark for at least 1 iterations and at least 0.5 seconds but terminate if exceeding 3600 seconds.\n"
    "INFO: count=1202 first=1938 curr=390 min=381 max=2081 avg=413.867 std=101 p5=386 median=390 p95=488\n\n"
    "INFO: Running benchmark for at least 20 iterations and at least 0 seconds but terminate if exceeding 3600 seconds.\n"
    "INFO: count=20 first=395 curr=386 min=385 max=426 avg=390.55 std=8 p5=386 median=388 p95=426\n\n"
)


def test_parse_real_output_ignores_the_warm_up_phase():
    res = parse_benchmark_output(REAL_STK_L21_FAST_DEFAULT)
    assert (res["avg"], res["max"], res["std"], res["runs"]) == (400341.0, 585000.0, 17000.0, 2487)
    # the warm-up phase had avg=483.029, max=8993, std=330, which the former parser reported


def test_parse_real_output_with_exactly_20_timed_runs():
    res = parse_benchmark_output(REAL_STK_L21_FAST_FIXED)
    assert (res["avg"], res["min"], res["max"], res["std"], res["runs"]) == (390550.0, 385000.0, 426000.0, 8000.0, 20)


def test_parse_reads_averages_in_scientific_notation():
    """benchmark_model prints averages of 1 s and more as e.g. 1.29563e+06 microseconds."""
    out = (WARMUP + "INFO: count=1 curr=1340000\n\n" + TIMED +
           "INFO: count=20 first=1320000 curr=1290000 min=1210000 max=1402000 avg=1.29563e+06 std=48210\n")
    res = parse_benchmark_output(out)
    assert res["avg"] == pytest.approx(1.29563e9)
    assert (res["min"], res["max"], res["runs"]) == (1.21e9, 1.402e9, 20)


def test_parse_timed_runs_that_all_took_the_same_time():
    out = WARMUP + "count=3 curr=500\n" + TIMED + "INFO: count=20 curr=500(all same)\n"
    assert not benchmark_failed(out)
    res = parse_benchmark_output(out)
    assert (res["avg"], res["min"], res["max"], res["std"], res["runs"]) == (500000.0, 500000.0, 500000.0, 0.0, 20)


def test_parse_reports_the_runs_actually_performed():
    assert parse_benchmark_output(WARMED_UP + "count=7 first=9 curr=9 min=8 max=10 avg=9 std=1\n")["runs"] == 7


@pytest.mark.parametrize("out", [
    "",
    "INFO: Loaded model\n",                                        # no benchmark run at all
    WARMUP + "count=9 first=5 curr=5 min=4 max=6 avg=5 std=1\n",   # warm-up only, timed runs missing
    TIMED + "count=20 first=5 curr=5 min=4 max=6 avg=5 std=1\n",   # timed runs without the warm-up phase
    WARMED_UP,                                                     # timed header but no statistics
    WARMED_UP + "count=20 first=5 curr=5 min=4 max=6\n",           # avg and std missing
    WARMED_UP + "count=20 first=5 curr=5 min=4 max=6 avg=9 std=1\n",   # average outside min..max
])
def test_parse_fails_explicitly_on_unusable_output(out):
    with pytest.raises(ValueError):
        parse_benchmark_output(out)


def test_benchmark_failed_detection():
    assert not benchmark_failed(OK_OUTPUT)
    assert benchmark_failed(GPU_FAIL_OUTPUT)
    assert benchmark_failed("")
    assert benchmark_failed("Failed to compute something\navg=1")
    assert benchmark_failed(WARMUP + "count=9 first=5 curr=5 min=4 max=6 avg=5 std=1\n")  # no timed runs


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
    assert (rec["iterations"], rec["cpu_runs"], rec["gpu_runs"], rec["npu_runs"]) == (20, 20, 20, 20)


def test_record_keeps_partial_failures():
    rec = build_record("AirNet", "Pixel", "13", 20,
                       {"cpu": ok(72000), "gpu": ok(10800), "npu": failed_result("NNAPI error")},
                       MEMORY, 32, ANALYTICS)
    assert rec["valid"] is True
    assert rec["unit"] == "GPU"
    assert rec["npu_duration"] == 0
    assert rec["npu_error"] == "NNAPI error"
    assert rec["npu_runs"] == 0


def test_record_all_backends_failed_is_invalid_but_kept():
    rec = build_record("Broken", "Dev", "10", 20,
                       {b: failed_result(f"{b} failed") for b in ("cpu", "gpu", "npu")},
                       MEMORY, 32, ANALYTICS)
    assert rec["valid"] is False
    assert "duration" not in rec and "unit" not in rec and "cpu_runs" not in rec
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
        "cpu_runs", "gpu_runs", "npu_runs",
        "npu_error",
    ]
    json.dumps(rec)  # must be serialisable


def test_record_takes_the_input_shape_of_any_model():
    rec = build_record("Custom", "Dev", "14", 20, {"cpu": ok(3), "gpu": ok(2), "npu": ok(4)},
                       MEMORY, (1, 1, 48, 40), ANALYTICS)
    assert (rec["in_dim_0"], rec["in_dim_1"], rec["in_dim_2"], rec["in_dim_3"]) == (1, 48, 40, 1)
