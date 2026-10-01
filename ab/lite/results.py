"""Parsing of benchmark_model output and construction of NN-Lite result records.

This module has no dependency on PyTorch, TensorFlow or adb, so it can be
unit-tested without a phone.
"""
import re

BACKENDS = ("cpu", "gpu", "npu")

ERROR_KEYWORDS = (
    "ERROR", "Error", "Failed", "Could not", "Aborted",
    "Cannot", "unsupported", "Unsupported", "INVALID",
    "Segmentation", "Op builtin_code", "Node number",
    "NNAPI", "GPU delegate", "Internal:", "INTERNAL:"
)


def extract_error_from_output(out: str) -> str:
    """Pull the real error message out of benchmark_model's stdout/stderr."""
    if not out or not out.strip():
        return "no output from benchmark_model"

    error_lines = []
    for line in out.splitlines():
        line = line.strip()
        if line and any(kw in line for kw in ERROR_KEYWORDS):
            error_lines.append(line)

    if error_lines:
        msg = " | ".join(error_lines[:3])
    else:
        nonempty = [ln.strip() for ln in out.splitlines() if ln.strip()]
        msg = nonempty[-1] if nonempty else "no output"

    if len(msg) > 500:
        msg = msg[:497] + "..."
    return msg


def benchmark_failed(out: str) -> bool:
    """True if benchmark_model reported an error or printed no statistics for the timed runs."""
    return "ERROR:" in out or "Failed to compute" in out or _timed_block(out) is None


def failed_result(error: str) -> dict:
    return {"avg": 0, "min": 0, "max": 0, "std": 0, "runs": 0, "status": "failed", "error": error}


# benchmark_model prints large averages in scientific notation, e.g. avg=1.29563e+06.
_NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_RUN_HEADER = "Running benchmark for at least"


def _timed_block(out: str):
    """The statistics line of the timed runs, or None.

    benchmark_model first runs a warm-up phase and then the timed runs, and prints a
    "Running benchmark for at least ..." header and a "count=..." statistics line for
    each phase. The timed runs are the second phase; output with only one phase (e.g.
    the phone disconnected during the warm-up) has no timed runs.
    """
    if out.count(_RUN_HEADER) < 2:
        return None
    match = re.search(r"count=\d+[^\n]*", out.rsplit(_RUN_HEADER, 1)[1])
    return match.group(0) if match else None


def parse_benchmark_output(out: str) -> dict:
    """Statistics of the timed runs, converted from microseconds to nanoseconds.

    Returns avg/min/max/std and the number of timed runs actually performed. The
    statistics of the warm-up phase are ignored. Raises ValueError if the output has
    no complete statistics for the timed runs, rather than returning plausible zeros.
    """
    line = _timed_block(out)
    if line is None:
        raise ValueError("no statistics for the timed runs in the benchmark_model output")
    runs = int(re.match(r"count=(\d+)", line).group(1))
    values = {key: re.search(rf"\b{key}=({_NUMBER})", line) for key in ("min", "max", "avg", "std")}
    if all(values.values()):
        stats = {key: float(match.group(1)) for key, match in values.items()}
    elif not any(values.values()) and (runs == 1 or "(all same)" in line) and re.search(rf"curr=({_NUMBER})", line):
        # Every run took the same time: benchmark_model prints only "count=N curr=T".
        t = float(re.search(rf"curr=({_NUMBER})", line).group(1))
        stats = {"min": t, "max": t, "avg": t, "std": 0.0}
    else:
        raise ValueError(f"incomplete statistics for the timed runs: {line.strip()[:120]}")
    if runs < 1 or not stats["min"] <= stats["avg"] <= stats["max"]:
        raise ValueError(f"inconsistent statistics for the timed runs: {line.strip()[:120]}")
    result = {key: value * 1000.0 for key, value in stats.items()}
    result.update(runs=runs, status="ok")
    return result


def build_record(model_name, device_model, os_version, iterations, results,
                 memory, input_size, device_analytics):
    """Build the JSON record for one model, precision and device.

    ``results`` maps "cpu"/"gpu"/"npu" to the dicts returned by
    ``parse_benchmark_output`` or ``failed_result``. ``input_size`` is the side of
    a square RGB input, or the model's NCHW input shape. The field order is part
    of the published schema and must not change.
    """
    if isinstance(input_size, int):
        input_size = (1, 3, input_size, input_size)
    batch, channels, height, width = input_size
    c, g, n = (results[b] for b in BACKENDS)
    opts = {}
    if c["status"] == "ok": opts["CPU"] = c["avg"]
    if g["status"] == "ok": opts["GPU"] = g["avg"]
    if n["status"] == "ok": opts["NPU"] = n["avg"]

    record = {
        "model_name": model_name,
        "device_type": device_model,
        "os_version": os_version,
        "valid": bool(opts),
        "emulator": False,
        "iterations": iterations,
    }
    if opts:
        winner = min(opts, key=opts.get)
        record.update({
            "duration": int(opts[winner]),
            "unit": winner,
            "cpu_duration": int(c["avg"]), "cpu_min_duration": int(c["min"]), "cpu_max_duration": int(c["max"]), "cpu_std_dev": c["std"],
            "gpu_duration": int(g["avg"]), "gpu_min_duration": int(g["min"]), "gpu_max_duration": int(g["max"]), "gpu_std_dev": g["std"],
            "npu_duration": int(n["avg"]), "npu_min_duration": int(n["min"]), "npu_max_duration": int(n["max"]), "npu_std_dev": n["std"],
        })
    record.update(memory)
    record.update({"in_dim_0": batch, "in_dim_1": height, "in_dim_2": width, "in_dim_3": channels,
                   "device_analytics": device_analytics})
    if opts:
        # Timed runs actually performed per backend (0 if it failed), next to the requested "iterations".
        record.update({f"{b}_runs": r.get("runs", 0) for b, r in zip(BACKENDS, (c, g, n))})
    for b, r in zip(BACKENDS, (c, g, n)):
        if r["status"] == "failed":
            record[f"{b}_error"] = r["error"]
    return record
