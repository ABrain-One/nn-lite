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
    """True if benchmark_model did not produce a usable timing summary."""
    return "ERROR:" in out or "Failed to compute" in out or "avg=" not in out.replace(" ", "")


def failed_result(error: str) -> dict:
    return {"avg": 0, "min": 0, "max": 0, "std": 0, "status": "failed", "error": error}


def parse_benchmark_output(out: str) -> dict:
    """Parse avg/min/max/std (microseconds in the tool output) into nanoseconds."""
    res = {"avg": 0.0, "min": 0.0, "max": 0.0, "std": 0.0, "status": "ok"}
    compact = out.replace(" ", "")
    for key in ["avg", "min", "max", "std"]:
        match = re.search(rf"{key}=([\d\.]+)", compact)
        if match:
            res[key] = float(match.group(1)) * 1000.0
    return res


def build_record(model_name, device_model, os_version, iterations, results,
                 memory, input_size, device_analytics):
    """Build the JSON record for one model, precision and device.

    ``results`` maps "cpu"/"gpu"/"npu" to the dicts returned by
    ``parse_benchmark_output`` or ``failed_result``. The field order is part of
    the published schema and must not change.
    """
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
    record.update({"in_dim_0": 1, "in_dim_1": input_size, "in_dim_2": input_size, "in_dim_3": 3,
                   "device_analytics": device_analytics})
    for b, r in zip(BACKENDS, (c, g, n)):
        if r["status"] == "failed":
            record[f"{b}_error"] = r["error"]
    return record
