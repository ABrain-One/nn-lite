#!/usr/bin/env python3
"""
torch2tflite.py - Standard version

Processes every model from nn-dataset/ab/nn/nn/ that has a matching .pth in
the HF source repo, except those already in processing_state_dual.json's
'processed' or 'failed' lists.

Use this version for normal full-coverage runs on a new device.
"""
import sys, os, argparse, json, re, subprocess, importlib.util, shutil, time, gc
from pathlib import Path

# --- CONFIGURATION ---
SOURCE_REPO = "NN-Dataset/checkpoints-epoch-50"
RESTART_EVERY_N_MODELS = 50 
COOL_DOWN_MODEL = 2        
COOL_DOWN_SESSION = 60     

# --- PATH SETUP ---
script_path = Path(__file__).resolve()
project_root = script_path.parents[3]
dataset_root = project_root / "nn-dataset"

stat_base = dataset_root / "ab" / "nn" / "stat" / "run" /  "tflite"
int8_dir = stat_base / "int8"
fp32_dir = stat_base / "fp32"

work_dir = dataset_root / "_work"
data_root, temp_dl_dir = work_dir / "data", work_dir / "temp"
state_file = work_dir / "processing_state_dual.json"

for p in [int8_dir, fp32_dir, data_root, temp_dl_dir]: 
    p.mkdir(parents=True, exist_ok=True)

import torch, torchvision, torchvision.transforms as T, ai_edge_torch, tensorflow as tf, numpy as np
from huggingface_hub import hf_hub_download, list_repo_files

# --- GUARDIAN: USB RECONNECTION LOGIC ---
def wait_for_device():
    print("\n[!] USB DISCONNECTED or DEVICE LOST. Waiting for reconnection...")
    while True:
        res = subprocess.run(["adb", "get-state"], capture_output=True, text=True)
        if "device" in res.stdout:
            print("[OK] Device detected. Re-initializing...")
            time.sleep(5)
            subprocess.run(["adb", "shell", "svc power stayon true"], capture_output=True)
            return
        time.sleep(10)

def adb_shell(cmd): 
    while True:
        res = subprocess.run(["adb", "shell", cmd], capture_output=True, text=True)
        if res.returncode != 0 and ("device not found" in res.stderr or "lost" in res.stderr):
            wait_for_device()
            continue
        return res.stdout.strip()

# --- BENCHMARK BINARY SETUP ---
def setup_benchmark_binary(force=False):
    """Push benchmark_model to device if not already present and executable."""
    if not force:
        check = adb_shell("ls /data/local/tmp/benchmark_model 2>/dev/null && echo OK")
        if "OK" in check:
            ver = adb_shell("/data/local/tmp/benchmark_model --version 2>&1 | head -1")
            if ver and "not found" not in ver.lower() and "permission denied" not in ver.lower():
                print(f"[SETUP] benchmark_model already on device: {ver}")
                return

    candidates = [
        script_path.parent / "benchmark_model",
        project_root / "benchmark_model",
        dataset_root / "benchmark_model",
        Path.cwd() / "benchmark_model",
    ]
    local_binary = next((c for c in candidates if c.exists()), None)
    if local_binary is None:
        raise FileNotFoundError(
            "benchmark_model binary not found. Place it next to this script "
            f"or in one of: {[str(c) for c in candidates]}"
        )

    print(f"[SETUP] Pushing {local_binary} to device...")
    pushed = False
    for attempt in range(3):
        res = subprocess.run(
            ["adb", "push", str(local_binary), "/data/local/tmp/"],
            capture_output=True, text=True,
        )
        if res.returncode == 0:
            pushed = True
            break
        if "device not found" in res.stderr or "lost" in res.stderr:
            wait_for_device()
            continue
        raise RuntimeError(f"adb push failed: {res.stderr}")
    if not pushed:
        raise RuntimeError("adb push failed after 3 attempts")

    print("[SETUP] Setting executable permission...")
    adb_shell("chmod +x /data/local/tmp/benchmark_model")
    ver = adb_shell("/data/local/tmp/benchmark_model --version 2>&1 | head -1")
    print(f"[SETUP] Verified: {ver}")

# --- METADATA HELPERS ---
def adb_getprop(key):
    lines = adb_shell(f"getprop {key}").splitlines()
    return lines[-1] if lines else ""

def get_gpu_name():
    raw = adb_shell("dumpsys SurfaceFlinger | grep GLES")
    if "Adreno" in raw:
        parts = raw.split(",")
        if len(parts) > 1: return parts[1].strip()
    return "Adreno (TM) 660"

def get_android_memory():
    mem = {}
    raw = adb_shell("cat /proc/meminfo")
    mapping = {"MemTotal": "total_ram_kb", "MemFree": "free_ram_kb", "MemAvailable": "available_ram_kb", "Cached": "cached_kb"}
    for line in raw.splitlines():
        parts = line.split(":")
        if len(parts) == 2:
            key = parts[0].strip()
            if key in mapping: mem[mapping[key]] = int(parts[1].strip().split()[0])
    return mem

def get_device_analytics():
    raw = adb_shell("cat /proc/cpuinfo")
    processors = []; current = {}
    global_meta = {"hardware": "", "features": "", "cpu implementer": "", "cpu architecture": "", "cpu variant": "", "cpu part": "", "cpu revision": ""}
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            if current: processors.append(current); current = {}
            continue
        if ":" in line: 
            k, v = line.split(":", 1); k, v = k.strip().lower(), v.strip()
            if k == "processor" and v.isdigit(): current["processor"] = v
            elif k in global_meta: global_meta[k] = v; current[k] = v
            else: current[k] = v
    if current: processors.append(current)
    soc = adb_getprop("ro.soc.model") or adb_getprop("ro.board.platform")
    return {"timestamp": time.time(), "cpu_info": {"cpu_cores": len([p for p in processors if 'processor' in p]), "processors": processors[:4], "arm_architecture": {"hardware": global_meta["hardware"] or soc, "features": global_meta["features"], "cpu_implementer": global_meta["cpu implementer"], "cpu_architecture": global_meta["cpu architecture"], "cpu_variant": global_meta["cpu variant"], "cpu_part": global_meta["cpu part"], "cpu_revision": global_meta["cpu revision"]}}}

# --- ERROR EXTRACTION ---
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

def run_bench(model_path, backend, runs, log_path=None, model_name=None, mode=None):
    """Run benchmark_model for one (backend, mode) combo and parse results."""
    flag = {"cpu": "--use_xnnpack=false", "gpu": "--use_gpu=true", "npu": "--use_nnapi=true"}.get(backend, "")    
    cmd = f"/data/local/tmp/benchmark_model --graph={model_path} --num_runs={runs} {flag}"
    out = adb_shell(cmd)

    if "ERROR:" in out or "Failed to compute" in out or "avg=" not in out.replace(" ", ""):
        error_msg = extract_error_from_output(out)
        if log_path is not None:
            try:
                with open(log_path, "a") as f:
                    f.write("=" * 72 + "\n")
                    f.write(f"timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"model:     {model_name}\n")
                    f.write(f"mode:      {mode}\n")
                    f.write(f"backend:   {backend}\n")
                    f.write(f"command:   {cmd}\n")
                    f.write(f"extracted: {error_msg}\n")
                    f.write("--- raw output ---\n")
                    f.write(out if out else "(empty)\n")
                    f.write("\n")
            except Exception as log_err:
                print(f"   [LOG WARN] could not write to {log_path}: {log_err}")

        return {"avg": 0, "min": 0, "max": 0, "std": 0, "status": "failed", "error": error_msg}

    res = {"avg": 0.0, "min": 0.0, "max": 0.0, "std": 0.0, "status": "ok"}
    for key in ["avg", "min", "max", "std"]:
        match = re.search(rf"{key}=([\d\.]+)", out.replace(" ", ""))
        if match: res[key] = float(match.group(1)) * 1000.0
    return res

# --- CORE LOGIC ---
def main():
    arch_dir, transforms_dir = dataset_root / "ab" / "nn" / "nn", dataset_root / "ab" / "nn" / "transform"
    ap = argparse.ArgumentParser()
    ap.add_argument("--android-runs", type=int, default=20)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--reinstall-bench", action="store_true",
                    help="Force re-push of benchmark_model binary to device")
    args = ap.parse_args()

    if args.force and state_file.exists(): state_file.unlink()
    state = json.load(open(state_file)) if state_file.exists() else {"processed": [], "failed": []}
    
    with open(hf_hub_download(SOURCE_REPO, "all_models.json", local_dir=str(work_dir))) as f: model_db = json.load(f)
    
    subprocess.run(["adb", "start-server"], capture_output=True)
    subprocess.run(["adb", "shell", "svc power stayon true"], capture_output=True)
    setup_benchmark_binary(force=args.reinstall_bench)
    
    gpu_full_name = get_gpu_name()
    device_model = adb_getprop("ro.product.model")
    device_clean = device_model.replace(" ", "_")
    os_ver = f"{adb_getprop('ro.build.version.release')} | {adb_getprop('ro.build.id')}"

    error_log = work_dir / f"benchmark_errors_{device_clean}.log"
    print(f"[LOG] Benchmark errors will be appended to: {error_log}")
    
    hf_files = list_repo_files(SOURCE_REPO)
    py_files = sorted([p for p in arch_dir.rglob("*.py") if f"{p.stem}.pth" in hf_files])
    to_process = [p for p in py_files
                  if p.stem not in set(state["processed"])
                  and p.stem not in set(state["failed"])]

    print(f"\n[DUAL RUN] Device: {device_model} | Remaining: {len(to_process)}")

    session_counter = 0
    for idx, py_path in enumerate(to_process, 1):
        name = py_path.stem
        time.sleep(COOL_DOWN_MODEL)
        
        if session_counter >= RESTART_EVERY_N_MODELS:
            print(f"\n[THERMAL] Resetting Session...")
            time.sleep(COOL_DOWN_SESSION)
            os.execv(sys.executable, [sys.executable, sys.argv[0]] + (["--android-runs", str(args.android_runs)] if "--android-runs" in sys.argv else []))

        print(f"\n[{idx}/{len(to_process)}] Model: {name}")
        try:
            prm = model_db[name].get("prm", {})
            target_h = 32
            tf_name = prm.get('transform', 'default') 
            tf_file = transforms_dir / f"{tf_name}.py"
            if tf_file.exists():
                match = re.search(r"(?:Resize|size|Crop).*?(\d+)", tf_file.read_text(), re.IGNORECASE)
                if match: 
                    target_h = int(match.group(1))
                    print(f"   [DEBUG] Transform: {tf_name} -> Res: {target_h}x{target_h}")
            else:
                print(f"   [DEBUG] Transform file {tf_name}.py not found. Defaulting to 32x32.")

            pth = Path(hf_hub_download(SOURCE_REPO, f"{name}.pth", cache_dir=str(temp_dl_dir)))
            spec = importlib.util.spec_from_file_location("mod", py_path)
            mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
            model = mod.Net((1,3,32,32), (10,), prm, "cpu")
            ckpt = torch.load(pth, map_location="cpu")
            model.load_state_dict(ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt, strict=False)
            model.eval()

            dummy_input = (torch.randn(1, 3, target_h, target_h),)

            # --- PROCESS FP32 ---
            print(f"   [PROCESS] FP32 Conversion...")
            fp32_tflite = temp_dl_dir / f"{name}_fp32.tflite"
            ai_edge_torch.convert(model, dummy_input).export(str(fp32_tflite))
            
            # --- PROCESS INT8 ---
            print(f"   [PROCESS] INT8 Conversion...")
            int8_tflite = temp_dl_dir / f"{name}_int8.tflite"
            int8_success = False
            
            try:
                def rep():
                    for j in range(50): 
                        yield [np.random.randn(1, 3, target_h, target_h).astype(np.float32)]
                
                ai_edge_torch.convert(
                    model, 
                    dummy_input, 
                    _ai_edge_converter_flags={
                        'optimizations': [tf.lite.Optimize.DEFAULT], 
                        'representative_dataset': rep, 
                        'target_spec': {'supported_ops': [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]}, 
                        'inference_input_type': tf.int8, 
                        'inference_output_type': tf.int8
                    }
                ).export(str(int8_tflite))
                int8_success = True

            except Exception as e_int8:
                print(f"   [WARN] INT8 Conversion failed: {e_int8}. Proceeding with FP32 benchmark only.")

            models_to_bench = [("fp32", fp32_tflite, fp32_dir)]
            if int8_success:
                models_to_bench.append(("int8", int8_tflite, int8_dir))

            for mode, tflite_path, save_dir in models_to_bench:
                dev_p = f"/data/local/tmp/{name}_{mode}.tflite"
                subprocess.run(["adb", "push", str(tflite_path), dev_p], capture_output=True)
                
                c = run_bench(dev_p, "cpu", args.android_runs, log_path=error_log, model_name=name, mode=mode)
                g = run_bench(dev_p, "gpu", args.android_runs, log_path=error_log, model_name=name, mode=mode)
                n = run_bench(dev_p, "npu", args.android_runs, log_path=error_log, model_name=name, mode=mode)
                adb_shell(f"rm {dev_p}")

                opts = {}
                if c["status"] == "ok": opts["CPU"] = c["avg"]
                if g["status"] == "ok": opts["GPU"] = g["avg"]
                if n["status"] == "ok": opts["NPU"] = n["avg"]

                # Original JSON field order preserved.
                if not opts:
                    print(f"   [INVALID] {mode.upper()}: all backends failed, marking valid=false")
                    final_data = {
                        "model_name": name,
                        "device_type": device_model,
                        "os_version": os_ver,
                        "valid": False,
                        "emulator": False,
                        "iterations": args.android_runs,
                        **get_android_memory(),
                        "in_dim_0": 1, "in_dim_1": target_h, "in_dim_2": target_h, "in_dim_3": 3,
                        "device_analytics": get_device_analytics()
                    }
                    if c["status"] == "failed": final_data["cpu_error"] = c["error"]
                    if g["status"] == "failed": final_data["gpu_error"] = g["error"]
                    if n["status"] == "failed": final_data["npu_error"] = n["error"]
                else:
                    winner = min(opts, key=opts.get)
                    final_data = {
                        "model_name": name,
                        "device_type": device_model,
                        "os_version": os_ver,
                        "valid": True,
                        "emulator": False,
                        "iterations": args.android_runs,
                        "duration": int(opts[winner]),
                        "unit": winner,
                        "cpu_duration": int(c["avg"]), "cpu_min_duration": int(c["min"]), "cpu_max_duration": int(c["max"]), "cpu_std_dev": c["std"],
                        "gpu_duration": int(g["avg"]), "gpu_min_duration": int(g["min"]), "gpu_max_duration": int(g["max"]), "gpu_std_dev": g["std"],
                        "npu_duration": int(n["avg"]), "npu_min_duration": int(n["min"]), "npu_max_duration": int(n["max"]), "npu_std_dev": n["std"],
                        **get_android_memory(),
                        "in_dim_0": 1, "in_dim_1": target_h, "in_dim_2": target_h, "in_dim_3": 3,
                        "device_analytics": get_device_analytics()
                    }
                    if c["status"] == "failed": final_data["cpu_error"] = c["error"]
                    if g["status"] == "failed": final_data["gpu_error"] = g["error"]
                    if n["status"] == "failed": final_data["npu_error"] = n["error"]

                model_folder = save_dir / f"img-classification_cifar-10_acc_{name}"
                model_folder.mkdir(parents=True, exist_ok=True)
                with open(model_folder / f"android_{device_clean}.json", "w") as f: 
                    json.dump(final_data, f, indent=2)

            print(f"   -> Successfully saved FP32 and INT8 stats.")
            state["processed"].append(name)
            json.dump(state, open(state_file, 'w'), indent=2)
            session_counter += 1
            gc.collect()
            if temp_dl_dir.exists(): shutil.rmtree(temp_dl_dir); temp_dl_dir.mkdir()
            
        except Exception as e:
            print(f"   [FAIL] {name}: {e}")
            state["failed"].append(name)
            json.dump(state, open(state_file, 'w'), indent=2)

if __name__ == "__main__": main()