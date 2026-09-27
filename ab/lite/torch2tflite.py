#!/usr/bin/env python3
"""
torch2tflite.py - Standard version

Processes every LEMUR model that has a matching .pth in the HF source repo,
except those already listed as processed or failed in the progress file of the
connected phone model (_work/processing_state_<model>.json). Phones are
identified by their model name, as in the result files.

Use this version for normal full-coverage runs on a new device.

Models are read from an nn-dataset checkout if one is given with --dataset-root
or NN_DATASET_ROOT, or found in an "nn-dataset" folder next to the nn-lite
checkout or in the current folder; results are then written into that checkout. Otherwise they are read
from the installed nn-dataset package and results are written to
./nn-lite-results. --out chooses another results folder in both cases.

With --model-path, models are read from local files instead (see
ab/lite/local_models.py) and the NN Dataset is not used at all.
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

def default_dataset_root():
    """The nn-dataset checkout from NN_DATASET_ROOT, next to the nn-lite checkout or in the
    current folder, or None when there is none and the installed nn-dataset package is used."""
    return find_checkout(None, os.environ.get("NN_DATASET_ROOT"),
                         project_root / "nn-dataset", Path.cwd() / "nn-dataset")

def set_dataset_root(root):
    """Point all output paths at the given nn-dataset checkout."""
    set_results_root(root)

def set_results_root(root):
    """Point all output paths at ``root``: an nn-dataset checkout or a results folder with the same layout."""
    global results_root, stat_base, int8_dir, fp32_dir, work_dir, data_root, temp_dl_dir
    results_root = Path(root).expanduser().resolve()
    stat_base = results_root / "ab" / "nn" / "stat" / "run" /  "tflite"
    int8_dir = stat_base / "int8"
    fp32_dir = stat_base / "fp32"
    work_dir = results_root / "_work"
    data_root, temp_dl_dir = work_dir / "data", work_dir / "temp"
    for p in [data_root, temp_dl_dir]:
        p.mkdir(parents=True, exist_ok=True)

try:
    from ab.lite.results import (extract_error_from_output, benchmark_failed, failed_result,
                                 parse_benchmark_output, build_record)
    from ab.lite.preprocess import CIFAR10_NORM, load_transform, calibration_images, input_size
    from ab.lite.progress import progress_file, load_progress, save_progress
    from ab.lite.sources import find_checkout, CheckoutSource, PackageSource
    from ab.lite.local_models import find_models, load_local_model, calibration_set, default_transform
except ImportError:  # executed as a plain script, e.g. after a session restart
    from results import (extract_error_from_output, benchmark_failed, failed_result,
                         parse_benchmark_output, build_record)
    from preprocess import CIFAR10_NORM, load_transform, calibration_images, input_size
    from progress import progress_file, load_progress, save_progress
    from sources import find_checkout, CheckoutSource, PackageSource
    from local_models import find_models, load_local_model, calibration_set, default_transform

import torch, torchvision, torchvision.transforms as T, litert_torch, tensorflow as tf, numpy as np
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
def setup_benchmark_binary(force=False, checkout=None):
    """Push benchmark_model to device if not already present and executable.

    The binary is looked for next to this script, in the nn-lite checkout, in the
    results folder, in the nn-dataset ``checkout`` if one is used, and in the
    current folder."""
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
        results_root / "benchmark_model",
        *([Path(checkout) / "benchmark_model"] if checkout else []),
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

# --- BENCHMARK ---
def run_bench(model_path, backend, runs, log_path=None, model_name=None, mode=None):
    """Run benchmark_model for one (backend, mode) combo and parse results."""
    flag = {"cpu": "--use_xnnpack=false", "gpu": "--use_gpu=true", "npu": "--use_nnapi=true"}.get(backend, "")    
    cmd = f"/data/local/tmp/benchmark_model --graph={model_path} --num_runs={runs} {flag}"
    out = adb_shell(cmd)

    if benchmark_failed(out):
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

        return failed_result(error_msg)

    return parse_benchmark_output(out)

# --- CORE LOGIC ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--android-runs", type=int, default=20)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--reinstall-bench", action="store_true",
                    help="Force re-push of benchmark_model binary to device")
    ap.add_argument("--dataset-root", default=None,
                    help="nn-dataset checkout to read models from and write results into "
                         "(default: $NN_DATASET_ROOT, or an nn-dataset folder next to the nn-lite checkout "
                         "or in the current folder; without one, the installed nn-dataset package is used)")
    ap.add_argument("--out", default=None,
                    help="Folder for the results (default: the nn-dataset checkout, or ./nn-lite-results "
                         "when the installed package is used)")
    ap.add_argument("--models", nargs="+", default=None,
                    help="Only process these model names (default: all models)")
    ap.add_argument("--model-path", "--model_path", nargs="+", default=None,
                    help="Benchmark models from these files or folders instead of the NN Dataset: "
                         ".pt2 files saved with torch.export, or .py files with a .pt/.pth file of the same name")
    ap.add_argument("--input-size", type=int, default=224,
                    help="Input side length for --model-path models given as .py and .pt files (default: 224)")
    ap.add_argument("--calib-dir", default=None,
                    help="Folder of sample images for INT8 calibration of --model-path models "
                         "(without it, only FP32 is benchmarked)")
    args = ap.parse_args()
    local = checkout = None
    if args.model_path:
        try:
            local, warnings = find_models(args.model_path)
            calib_set = calibration_set(args.calib_dir) if args.calib_dir else None
        except ValueError as e:
            ap.error(str(e))
        for w in warnings: print(f"[WARN] {w}")
        set_results_root(args.out or "nn-lite-results")
        print(f"[SETUP] {len(local)} model(s) from --model-path; the NN Dataset is not used")
    else:
        try:
            checkout = find_checkout(args.dataset_root, os.environ.get("NN_DATASET_ROOT"),
                                     project_root / "nn-dataset", Path.cwd() / "nn-dataset")
        except ValueError as e:
            ap.error(str(e))
        set_results_root(args.out or checkout or "nn-lite-results")
        if checkout:
            print(f"[SETUP] Models are read from the nn-dataset checkout {checkout}")
            source = CheckoutSource(checkout)
        else:
            print("[SETUP] Models are read from the installed nn-dataset package")
            if not (work_dir / "lemur" / "db" / "ab.nn.db").exists():
                print("[SETUP] Downloading the LEMUR database (about 1.2 GB unpacked); this happens only once")
            source = PackageSource.open(work_dir / "lemur", work_dir / "code")
    print(f"[SETUP] Results are written to {results_root}")

    if not local:
        with open(hf_hub_download(SOURCE_REPO, "all_models.json", local_dir=str(work_dir))) as f: model_db = json.load(f)

    subprocess.run(["adb", "start-server"], capture_output=True)
    subprocess.run(["adb", "shell", "svc power stayon true"], capture_output=True)
    setup_benchmark_binary(force=args.reinstall_bench, checkout=checkout)

    gpu_full_name = get_gpu_name()
    device_model = adb_getprop("ro.product.model")
    device_clean = device_model.replace(" ", "_")
    os_ver = f"{adb_getprop('ro.build.version.release')} | {adb_getprop('ro.build.id')}"

    error_log = work_dir / f"benchmark_errors_{device_clean}.log"
    print(f"[LOG] Benchmark errors will be appended to: {error_log}")

    # Progress is kept per phone, so a new phone starts from the beginning.
    state_file = progress_file(work_dir, device_clean + ("_custom" if local else ""))
    if args.force and state_file.exists(): state_file.unlink()
    state = load_progress(state_file)
    print(f"[LOG] Progress for this phone: {state_file}")

    if local:
        names = sorted(local)
    else:
        # INT8 calibration uses real CIFAR-10 training images, preprocessed with each model's own transform.
        calib_set = torchvision.datasets.CIFAR10(root=str(data_root), train=True, download=True)
        hf_files = list_repo_files(SOURCE_REPO)
        names = sorted(n for n in source.names() if f"{n}.pth" in hf_files)
    if args.models:
        unknown = sorted(set(args.models) - set(names))
        if unknown:
            print(f"[WARN] No model code or checkpoint for: {', '.join(unknown)}")
        names = [n for n in names if n in set(args.models)]
    to_process = [n for n in names
                  if n not in set(state["processed"])
                  and n not in set(state["failed"])]

    print(f"\n[DUAL RUN] Device: {device_model} | Remaining: {len(to_process)}")

    session_counter = 0
    for idx, name in enumerate(to_process, 1):
        time.sleep(COOL_DOWN_MODEL)
        
        if session_counter >= RESTART_EVERY_N_MODELS:
            print(f"\n[THERMAL] Resetting Session...")
            time.sleep(COOL_DOWN_SESSION)
            restart_args = ["--android-runs", str(args.android_runs), "--out", str(results_root)]
            if checkout: restart_args += ["--dataset-root", str(checkout)]
            if local:
                restart_args += ["--model-path", *args.model_path, "--input-size", str(args.input_size)]
                if args.calib_dir: restart_args += ["--calib-dir", args.calib_dir]
            if args.models: restart_args += ["--models", *args.models]
            os.execv(sys.executable, [sys.executable, sys.argv[0]] + restart_args)

        print(f"\n[{idx}/{len(to_process)}] Model: {name}")
        try:
            if local:
                model, shape, model_tf = load_local_model(local[name], args.input_size)
                model_tf = model_tf or default_transform(shape)
                print(f"   [DEBUG] Input: {'x'.join(map(str, shape))}")
            else:
                prm = model_db[name].get("prm", {})
                tf_name = prm.get('transform', 'default')
                tf_file = source.transform_file(tf_name)
                if tf_file:
                    model_tf = load_transform(tf_file.parent, tf_name)
                    try:
                        # The input size is what the model's own transform actually produces.
                        target_h = input_size(model_tf)
                    except Exception:
                        # Transforms that cannot be applied to a plain image (e.g. detection or
                        # super-resolution pipelines): fall back to the first number in the source.
                        match = re.search(r"(?:Resize|size|Crop).*?(\d+)", tf_file.read_text(), re.IGNORECASE)
                        target_h = int(match.group(1)) if match else 32
                    print(f"   [DEBUG] Transform: {tf_name} -> Res: {target_h}x{target_h}")
                else:
                    model_tf = T.Compose([T.ToTensor(), T.Normalize(*CIFAR10_NORM)])
                    target_h = 32
                    print(f"   [DEBUG] Transform file {tf_name}.py not found. Defaulting to 32x32.")

                pth = Path(hf_hub_download(SOURCE_REPO, f"{name}.pth", cache_dir=str(temp_dl_dir)))
                spec = importlib.util.spec_from_file_location("mod", source.model_file(name))
                mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
                # Build the model for the input size it was trained with, as nn-dataset does.
                model = mod.Net((1,3,target_h,target_h), (10,), prm, "cpu")
                ckpt = torch.load(pth, map_location="cpu")
                model.load_state_dict(ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt, strict=False)
                model.eval()
                shape = (1, 3, target_h, target_h)

            dummy_input = (torch.randn(*shape),)

            # --- PROCESS FP32 ---
            print(f"   [PROCESS] FP32 Conversion...")
            fp32_tflite = temp_dl_dir / f"{name}_fp32.tflite"
            litert_torch.convert(model, dummy_input).export(str(fp32_tflite))
            
            # --- PROCESS INT8 ---
            int8_tflite = temp_dl_dir / f"{name}_int8.tflite"
            int8_success = False
            if calib_set is None:
                print(f"   [INFO] INT8 skipped: give --calib-dir with sample images to calibrate it.")
            else:
                print(f"   [PROCESS] INT8 Conversion...")
                try:
                    calib = calibration_images(calib_set, model_tf, shape[2:])
                    def rep():
                        for j in range(len(calib)):
                            yield [calib[j:j + 1]]
                
                    litert_torch.convert(
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

                memory = get_android_memory()
                final_data = build_record(name, device_model, os_ver, args.android_runs,
                                          {"cpu": c, "gpu": g, "npu": n}, memory, shape,
                                          get_device_analytics())
                if not final_data["valid"]:
                    print(f"   [INVALID] {mode.upper()}: all backends failed, marking valid=false")

                if local:
                    model_folder = results_root / "custom" / mode / name
                else:
                    model_folder = save_dir / f"img-classification_cifar-10_acc_{name}"
                model_folder.mkdir(parents=True, exist_ok=True)
                with open(model_folder / f"android_{device_clean}.json", "w") as f: 
                    json.dump(final_data, f, indent=2)

            print(f"   -> Successfully saved {' and '.join(m.upper() for m, _, _ in models_to_bench)} stats.")
            state["processed"].append(name)
            save_progress(state_file, state)
            session_counter += 1
            gc.collect()
            if temp_dl_dir.exists(): shutil.rmtree(temp_dl_dir); temp_dl_dir.mkdir()
            
        except Exception as e:
            print(f"   [FAIL] {name}: {e}")
            state["failed"].append(name)
            save_progress(state_file, state)

if __name__ == "__main__": main()