# Changelog

All notable changes to NN-Lite are documented in this file.

## [1.0.0] - 2026-09-30

### Added
- Physical-device benchmarking engine (`ab/lite/torch2tflite.py`): FP32 and INT8 conversion,
  CPU, GPU and NNAPI timing with the LiteRT `benchmark_model` binary over adb, automatic
  push of the binary to the phone, USB reconnection, thermal cool-downs, periodic restarts and
  a resumable processing ledger.
- Per-backend error extraction into the JSON records and a per-device error log; records in
  which every backend fails are kept with `valid: false`.
- `pyproject.toml` for installation with pip and the `nn-lite-bench` command.
- `--dataset-root` option and `NN_DATASET_ROOT` environment variable to locate the NN Dataset
  checkout; `--models` option to benchmark selected models only.
- Unit tests for output parsing, error extraction and the record schema, with GitHub Actions CI.
- `CONTRIBUTING.md` with support and governance information.
- `--model-path` option to benchmark models from local files, without the NN Dataset
  (`ab/lite/local_models.py`): `.pt2` files saved with `torch.export`, or a `.py` file with a
  `.pt`/`.pth` file of the same name holding the weights or the whole model. INT8 is calibrated
  with images given with `--calib-dir`; results are written to `nn-lite-results/custom/`.
- `--serial` option (default `$ANDROID_SERIAL`) to choose the phone when several are
  connected, so several phones can be benchmarked at the same time, one run per phone
  (`ab/lite/adb.py`). Every adb command is sent to the chosen phone with `adb -s`; without
  `--serial`, a run with several phones connected stops and lists them. Runs for phones of
  different models can share a results folder: each phone has its own temporary folder, and
  the shared downloads and each phone model's progress ledger are locked (`ab/lite/locks.py`).
- `--restart-every N` option to set how often the process restarts to free memory (default 50
  models, as before; 0 never restarts), so the restart can be tried in a few minutes.
- JOSS paper draft in `doc/joss-paper/`.

### Changed
- The NN Dataset package is now a dependency, and models and transforms are read from it by
  default (`ab/lite/sources.py`), so `pip install nn-lit` is all that is needed. The LEMUR
  database is downloaded on first use; results are written to `./nn-lite-results` (or `--out`)
  in the dataset's layout. An nn-dataset checkout given with `--dataset-root` or
  `NN_DATASET_ROOT`, or found next to a source checkout of NN-Lite or in the current folder, is
  still read from and written into as before.
- `ab/__init__.py` extends its search path, so `ab.lite` from a source checkout and `ab.nn` from
  the installed NN Dataset package can be imported together.
- Conversion now uses `litert-torch` (formerly `ai-edge-torch`), including in the emulator path.
- Result parsing and record construction moved to `ab/lite/results.py`; the record format is
  unchanged.
- README restructured around installation, connecting a phone and a worked example.
- Progress is stored per phone model (`_work/processing_state_<model>.json`,
  `ab/lite/progress.py`) instead of in one shared file, so a phone of another model starts from
  the beginning and `--force` only resets the connected phone model.
- `requirements.txt` now lists only direct dependencies, pinned to stable releases where they
  exist (torch 2.9.1, torchvision 0.24.1, ai-edge-litert 2.1.2, torchao 0.15.0); nightly
  packages were removed. The previous pin `ai-edge-litert-nightly==2.2.0.dev20260202` is no
  longer available on PyPI, so fresh installs had failed.
- ai-edge-litert 2.1.2 with ai-edge-quantizer 0.4.2 instead of 2.0.3 with 0.4.0: the macOS
  build of ai-edge-litert 2.0.3 lacks `libpywrap_litert_common.dylib`, so the converter
  could not be imported on Apple Silicon Macs.

### Fixed
- Models are built for the input size they were trained with, as in the NN Dataset, instead of
  always for 32x32 inputs; this matters for architectures whose layers depend on the input size.
- A `--dataset-root` or `NN_DATASET_ROOT` that is not an nn-dataset checkout is reported as an
  error; previously the run found no models and ended with "Remaining: 0".
- The input size of each model is taken from the output of its own transform. It was read as the
  first number in the transform's source, which is wrong for transforms such as
  `CenterCrop(32)` followed by `Resize((64, 64))`, so such models were benchmarked at the wrong
  resolution.
- INT8 calibration uses 50 real CIFAR-10 training images, preprocessed with each model's own
  transform (`ab/lite/preprocess.py`), instead of random noise. Random-noise calibration, used
  since May 2026, lowered the INT8 accuracy of some models considerably (e.g. AirNet on
  CIFAR-10: 79.1% instead of 86.6%).
- Crash on devices with missing system properties (e.g. HiSilicon Kirin).
- The emulator path no longer imports the removed `ai_edge_torch` package.

## Earlier development (2025)

- Emulator-based pipeline (`ab/lite/torch2tflite-all.py`) with Android Virtual Device
  management, the NN-Lite Android app for CPU/GPU/NNAPI inference, crash recovery and device
  analytics.
