# NN-Lite API reference

This document describes the public interface of NN-Lite 1.0.0: the command-line
tool, the Python modules that can be imported by other code, and the format of
the records the pipeline writes.

Anything documented here is covered by the compatibility promise in
[CONTRIBUTING.md](../CONTRIBUTING.md): field names, field meanings and field
order in the JSON records do not change within a major version; new fields may be
added. Names not listed here are internal and may change at any time.

---

## 1. Command-line interface

### `nn-lite-bench`

Benchmarks models on an Android phone attached over USB, either from
local files given with `--model-path` or from the NN Dataset (LEMUR). Dataset
models are read from an `nn-dataset` checkout if one is found (see below),
otherwise from the `nn-dataset` package that is installed together with
NN-Lite. Installed by `pip install nn-lit`. From a source checkout
the equivalent command is `python -m ab.lite.torch2tflite`.

```bash
nn-lite-bench [--models NAME [NAME ...]] [--android-runs N] [--serial SERIAL] [--restart-every N]
              [--dataset-root PATH] [--out PATH] [--force] [--reinstall-bench]
nn-lite-bench --model-path PATH [PATH ...] [--input-size N] [--calib-dir DIR]
              [--models NAME [NAME ...]] [--android-runs N] [--serial SERIAL] [--restart-every N]
              [--out PATH] [--force]
```

| Option | Default | Meaning |
|---|---|---|
| `--models NAME [NAME ...]` | all models | Benchmark only the named models. Names are the model file stems in `ab/nn/nn/`, e.g. `AirNet`, or the file stems of the `--model-path` models. |
| `--android-runs N` | `20` | Timed runs per backend, passed to `benchmark_model --num_runs`. |
| `--dataset-root PATH` | see below | Location of the `nn-dataset` checkout that models are read from and records are written to. An error if `PATH` is not a checkout. |
| `--out PATH` | the checkout, or `./nn-lite-results` | Folder the records and working files are written to. |
| `--serial SERIAL` | `$ANDROID_SERIAL`, or the only phone connected | The phone to benchmark, by the serial number `adb devices` lists for it. Needed when several phones are connected. |
| `--restart-every N` | `50` | After every N models, pause for 60 s and restart the process, which releases the memory the converter keeps between models; the new process continues from the progress ledger with the same options, except `--force` and `--reinstall-bench`. `0` never restarts. |
| `--force` | off | Delete the progress ledger of the connected phone and start from the beginning. Other phones are unaffected. |
| `--reinstall-bench` | off | Push the `benchmark_model` binary to the phone again, even if it is already present. |
| `--model-path PATH [PATH ...]` | off | Benchmark models from local files or folders instead of the NN Dataset (see below). `--model_path` is accepted too. |
| `--input-size N` | `224` | Square input size for `--model-path` models given as `.py` and `.pt`/`.pth` files. |
| `--calib-dir DIR` | none | Images for INT8 calibration of `--model-path` models; without it only FP32 is benchmarked. |

The phone must be authorised (`adb devices` must list it in state `device`).
Without `--serial`, exactly one phone may be in that state; with several, the run
stops and lists their serial numbers. Every adb command of a run is sent to its
phone with `adb -s <serial>`. A `--serial` that `adb devices` does not list is
waited for, as after a USB disconnection.

Several phones are benchmarked at the same time by starting one run per phone,
each with its own `--serial`. The runs may use the same results folder, except
that two phones of the same model (the same `ro.product.model`) cannot be
benchmarked at the same time: they share a progress ledger and result files, so
the second run stops with an error.

### Locating the dataset

The dataset root is resolved in this order:

1. `--dataset-root PATH`
2. the `NN_DATASET_ROOT` environment variable
3. a sibling `nn-dataset` directory next to the `nn-lite` checkout
4. `./nn-dataset` in the current working directory
5. otherwise the installed `nn-dataset` package: the first run downloads the LEMUR
   database (about 1.2 GB unpacked) from Hugging Face, and records are written to
   `./nn-lite-results` in the same layout as a checkout

Rules 3 and 4 apply only to directories that are checkouts (contain `ab/nn/nn/`).
The first lines of the output say which source and results folder are used.

```bash
export NN_DATASET_ROOT=/data/nn-dataset
nn-lite-bench --models AirNet
```

### Models from local files

With `--model-path`, the NN Dataset is not used at all. Each path is a file or a
folder of files:

- `<name>.pt2`: a model saved with `torch.export.save`. No Python code is needed;
  the input shape is read from the file. The model should be exported in
  evaluation mode, as the exported graph keeps the mode it was exported in.
- `<name>.py` with `<name>.pt` or `<name>.pth` next to it: the model's code and
  either its `state_dict` (also inside a checkpoint dictionary under `state_dict`,
  `model_state_dict` or `model`) or the whole model saved with `torch.save(model)`.
  The network is created with `create_model()` if the `.py` file defines it,
  otherwise with `Net()` or the file's only `nn.Module` subclass. An optional
  `input_transform(image)` turns a PIL image into a CHW tensor for calibration;
  by default images are resized and normalised with the ImageNet statistics.

A `.pt`/`.pth` file without its `.py` file is an error when given directly and a
warning inside a folder, as it cannot be converted on its own. Only models with a
single 4-D (NCHW) input are supported.

```bash
nn-lite-bench --model-path my_models/ --calib-dir sample_images/
```

### Emulator path

```bash
python -m ab.lite.torch2tflite-all [NAME ...]
```

Runs models inside an Android emulator through the app in `App/`. It requires
Android Studio and the `nn-dataset` Python package, which is installed with NN-Lite
(the `emulator` extra, `pip install nn-lit[emulator]`, is kept for compatibility),
and writes records with `emulator: true` in the same schema as the physical-device
path, so results from both can be stored and filtered together. They are saved
in `out/benchmark_reports/<task>_<Model>/` rather than in `ab/nn/stat/`. `out/` is
created by `nn-dataset` in the first directory, from the current one upwards,
that contains both an `ab` folder and a `README.md` (such as an `nn-dataset`
checkout), or otherwise in the current directory.

---

## 2. Files written

For each model, precision and device, where `<results>` is the dataset root, the
`--out` folder or `./nn-lite-results`:

```
<results>/ab/nn/stat/run/tflite/fp32/img-classification_cifar-10_acc_<Model>/android_<device>.json
<results>/ab/nn/stat/run/tflite/int8/img-classification_cifar-10_acc_<Model>/android_<device>.json
```

Because the layout is the same, the `ab/` folder of `./nn-lite-results` can be
copied into an `nn-dataset` checkout and committed.

Models given with `--model-path` are written to
`<results>/custom/{fp32,int8}/<name>/android_<device>.json`, where `<results>` is
the `--out` folder or `./nn-lite-results`.

Working files, all under `<results>/_work/`:

| Path | Contents |
|---|---|
| `processing_state_<device>.json` | Progress ledger for one phone: `{"processed": [...], "failed": [...]}`. |
| `benchmark_errors_<device>.log` | Full `benchmark_model` output for every failed backend invocation. |
| `data/` | CIFAR-10 download used for INT8 calibration. |
| `temp_<serial>/` | Checkpoint downloads and converted models of one phone, cleared after each model. |
| `downloads.lock`, `processing_state_<device>.json.lock` | Locks between runs for different phones (see `ab.lite.locks`). |
| `lemur/` | LEMUR database of the installed `nn-dataset` package (only when no checkout is used). |
| `processing_state_<device>_custom.json` | Progress ledger for `--model-path` models, separate from the dataset's. |
| `code/` | Model and transform code taken from that database. |

`<device>` is the phone's `ro.product.model` with spaces replaced by `_`; in the
name of the progress ledger, every character outside `[A-Za-z0-9._-]` is replaced
by `_`.

---

## 3. Record schema

One JSON object per model, precision and device. The field order below is part of
the schema.

| Field | Type | Meaning |
|---|---|---|
| `model_name` | string | Model file stem, e.g. `AirNet`. |
| `device_type` | string | `ro.product.model` of the phone. |
| `os_version` | string | Android release and build, e.g. `10 \| HUAWEISTK-L21`. |
| `valid` | bool | `false` if every backend failed; the record is still written. |
| `emulator` | bool | `false` for the physical-device path. |
| `iterations` | int | Timed runs requested per backend (`--android-runs`). |
| `duration` | int | Latency of the fastest backend, nanoseconds. Absent when `valid` is `false`. |
| `unit` | string | Fastest backend: `CPU`, `GPU` or `NPU`. Absent when `valid` is `false`. |
| `cpu_duration`, `cpu_min_duration`, `cpu_max_duration` | int | Mean, minimum and maximum CPU latency, nanoseconds. |
| `cpu_std_dev` | float | Standard deviation of CPU latency, nanoseconds. |
| `gpu_*`, `npu_*` | | The same four fields for the GPU and NPU backends. |
| `total_ram_kb`, `free_ram_kb`, `available_ram_kb`, `cached_kb` | int | Phone memory state at measurement time. |
| `in_dim_0` … `in_dim_3` | int | Input shape as batch, height, width, channels. |
| `device_analytics` | object | Core count, CPU implementer/architecture/variant/part/revision, features and SoC string. |
| `cpu_runs`, `gpu_runs`, `npu_runs` | int | Timed runs actually performed per backend, as reported by `benchmark_model`; `0` for a failed backend. Absent when `valid` is `false`. Added in 1.0.1. |
| `cpu_error`, `gpu_error`, `npu_error` | string | Present only for a backend that failed; up to three extracted error lines, truncated to 500 characters. |

When at least one backend succeeded, a failed backend still has its four numeric
fields, set to `0`; the presence of the matching `*_error` key is what marks it as
failed. When every backend failed, the record has no latency fields at all, only
the `*_error` keys. All latencies are
nanoseconds — `benchmark_model` reports microseconds, which NN-Lite converts on
parse.

The latencies are those of the timed runs only. `benchmark_model` first runs a
warm-up phase and reports statistics for it too; NN-Lite ignores them. It is run
with `--num_runs=<iterations> --min_secs=0 --max_secs=3600`, so every backend
performs exactly `iterations` timed runs: by default, `benchmark_model` would also
repeat a model until one second had passed, and stop after 150 seconds.

Example (abridged):

```json
{
  "model_name": "AirNet",
  "device_type": "STK-L21",
  "os_version": "10 | HUAWEISTK-L21",
  "valid": true,
  "emulator": false,
  "iterations": 20,
  "duration": 55200000,
  "unit": "GPU",
  "cpu_duration": 329480000, "cpu_min_duration": 310111000,
  "cpu_max_duration": 344513000, "cpu_std_dev": 9657000.0,
  "gpu_duration": 55200000, "gpu_min_duration": 53553000,
  "gpu_max_duration": 60863000, "gpu_std_dev": 2064000.0,
  "npu_duration": 364514000, "npu_min_duration": 357856000,
  "npu_max_duration": 371171000, "npu_std_dev": 6657000.0,
  "total_ram_kb": 3775716, "free_ram_kb": 189496,
  "available_ram_kb": 1617764, "cached_kb": 1638264,
  "in_dim_0": 1, "in_dim_1": 128, "in_dim_2": 128, "in_dim_3": 3,
  "device_analytics": {"timestamp": 1758979200.0, "cpu_info": {"cpu_cores": 8}},
  "cpu_runs": 20, "gpu_runs": 20, "npu_runs": 20
}
```

---

## 4. Python API

### `ab.lite.results` — parsing and records

No dependency on PyTorch, TensorFlow or `adb`; importable and testable anywhere.

```python
from ab.lite.results import (
    BACKENDS, parse_benchmark_output, benchmark_failed,
    extract_error_from_output, failed_result, build_record,
)
```

**`BACKENDS`** — `("cpu", "gpu", "npu")`, the canonical order.

**`parse_benchmark_output(out: str) -> dict`**
Parses the statistics of the timed runs — the second "Running benchmark for at
least …" phase; the warm-up phase before it is ignored — and returns
`{"avg", "min", "max", "std", "runs", "status"}`, with latencies converted from
microseconds to nanoseconds, `runs` the number of timed runs performed and
`status` set to `"ok"`. Averages in scientific notation (`avg=1.29563e+06`) and
the `count=N curr=T (all same)` form are supported. Raises `ValueError` if the
statistics of the timed runs are missing, incomplete or inconsistent, instead of
returning plausible zeros.

**`benchmark_failed(out: str) -> bool`**
`True` when the output contains no usable timing summary — an `ERROR:` line, a
compute failure, or no statistics for the timed runs.

**`extract_error_from_output(out: str) -> str`**
Returns a one-line diagnosis: up to the first three lines matching the module's
error keywords, joined by `" | "` and truncated to 500 characters. Falls back to
the last non-empty line, or `"no output from benchmark_model"` for empty output.

**`failed_result(error: str) -> dict`**
A zeroed result dict with `status="failed"` and the given message, in the shape
`build_record` expects.

**`build_record(model_name, device_model, os_version, iterations, results, memory, input_size, device_analytics) -> dict`**
Assembles one record. `results` maps `"cpu"`, `"gpu"` and `"npu"` to dicts from
`parse_benchmark_output` or `failed_result`; `memory` is a dict of the four
`*_kb` fields; `input_size` is the square input edge of an RGB model, or the
model's input shape as an `(N, C, H, W)` tuple. The fastest successful
backend becomes `duration`/`unit`; if none succeeded, `valid` is `False` and
those two fields are omitted. The returned dict is JSON-serialisable and its key
order is the published schema.

```python
rec = build_record(
    "AirNet", "SM-F926B", "14 | UP1A", 20,
    {"cpu": parse_benchmark_output(cpu_out),
     "gpu": parse_benchmark_output(gpu_out),
     "npu": failed_result("NNAPI delegate unavailable")},
    {"total_ram_kb": 11631760, "free_ram_kb": 2818008,
     "available_ram_kb": 5356556, "cached_kb": 4672460},
    128, {"timestamp": 0.0, "cpu_info": {"cpu_cores": 8}},
)
```

### `ab.lite.preprocess` — calibration inputs

```python
from ab.lite.preprocess import CIFAR10_NORM, load_transform, preprocess, calibration_images
```

**`CIFAR10_NORM`** — the CIFAR-10 mean/standard-deviation pair used by NN Dataset.

**`load_transform(transforms_dir, name, norm=CIFAR10_NORM) -> Callable`**
Loads the torchvision transform `name` from `<dataset-root>/ab/nn/transform/`
and instantiates it with `norm`. Each model declares the transform it was trained
with, so calibration reproduces its training inputs rather than one fixed
normalisation.

**`preprocess(images, transform, size) -> np.ndarray`**
Applies `transform` to PIL images and returns a `float32` NCHW array, bilinearly
resized to `size × size` (or to `size = (height, width)`) if the transform's
output differs.

**`calibration_images(dataset, transform, size, count=50) -> np.ndarray`**
The first `count` images (or all, if there are fewer) of an `(image, label)`
dataset, preprocessed for quantization. Used as the representative dataset for full-integer post-training
quantization.

### `ab.lite.progress` — resumable runs

```python
from ab.lite.progress import progress_file, load_progress, save_progress
```

**`progress_file(work_dir, device) -> Path`** — path of the ledger for one phone;
the device string is sanitised into the filename.
**`load_progress(path) -> dict`** — the ledger, or `{"processed": [], "failed": []}`
if it does not exist.
**`save_progress(path, state)`** — writes the ledger as indented JSON.

Ledgers are per phone, so benchmarking a second device does not skip models
measured only on the first, and `--force` resets one device only.

### `ab.lite.adb` — the phone in use

```python
from ab.lite import adb
```

**`adb.serial`** — serial number of the phone every command is sent to (`None` for
adb's default phone).
**`command(*args) -> list`** / **`run(*args) -> CompletedProcess`** — the adb
command line for `args`, with `-s <serial>`, and running it.
**`list_devices(output) -> dict`** — `{serial: state}` from the output of
`adb devices`; **`connected_devices()`** runs it.
**`choose_device(requested, devices) -> str`** — the serial to use: `requested`
if given, otherwise the only phone in state `device`; raises `ValueError`
otherwise.

### `ab.lite.locks` — several phones, one results folder

**`downloading(work_dir)`** — context manager held while downloading the files
the runs share; waits while another run holds it.
**`claim(progress_path, device_model)`** — locks a phone model's progress ledger
until the run ends; raises `RuntimeError` if another run holds it.

### `ab.lite.options` — command-line options and restarts

**`parser()`** — the argument parser of `nn-lite-bench` with the options above.
**`restart_args(args, results_root, serial, checkout=None) -> list`** — the options
the process restarts itself with after every `--restart-every` models: the same
options, results folder and phone, without `--force` and `--reinstall-bench`.

### `ab.lite.torch2tflite` — the pipeline

**`main()`** — entry point behind `nn-lite-bench`; parses the arguments above and
runs the benchmarking loop.
**`default_dataset_root() -> Path | None`** — applies rules 2–4 of the resolution
order; `None` when no checkout is found and the installed package is used.
**`set_dataset_root(root)`** — points all output paths at a checkout (or any
results folder) and creates the directories it needs. Call before other functions
in this module.
**`run_bench(model_path, backend, runs, log_path=None, model_name=None, mode=None) -> dict`**
— runs `benchmark_model` on the phone for one backend and returns a parsed result
or `failed_result`, appending failures to `log_path`.

---

## 5. Extending NN-Lite

Consuming records without running a phone:

```python
import json, glob
results = "nn-lite-results"          # or the nn-dataset checkout that was written to
for pattern in (f"{results}/ab/nn/stat/run/tflite/*/*/android_*.json",
                f"{results}/custom/*/*/android_*.json"):
    for path in glob.glob(pattern):
        rec = json.load(open(path))
        if rec["valid"]:
            print(rec["model_name"], rec["unit"], rec["duration"] / 1e6, "ms")
```

Producing records from another measurement source: build the per-backend dicts
yourself and call `build_record`, which keeps the schema consistent with the rest
of the dataset.

Adding a backend: extend `BACKENDS` and the flag map in `run_bench`, add the four
numeric fields and the `*_error` field to `build_record` **after** the existing
ones, and extend the schema test in `tests/test_results.py`.

---

## 6. Running the tests

The unit tests cover parsing, error extraction, the record schema, calibration
preprocessing, the progress ledger, the choice of the model source, finding
`--model-path` models, choosing the phone, the locks between runs and the options
kept across restarts. They need neither a phone nor PyTorch, TensorFlow or the NN
Dataset; the tests that load `--model-path` models run when PyTorch is installed:

```bash
pip install pytest filelock
python -m pytest tests
```
