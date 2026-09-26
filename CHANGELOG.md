# Changelog

All notable changes to NN-Lite are documented in this file.

## [1.0.0] - Unreleased

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
- JOSS paper draft in `paper/`.

### Changed
- Conversion now uses `litert-torch` (formerly `ai-edge-torch`), including in the emulator path.
- Result parsing and record construction moved to `ab/lite/results.py`; the record format is
  unchanged.
- README restructured around installation, connecting a phone and a worked example.
- `requirements.txt` now lists only direct dependencies, pinned to stable releases where they
  exist (torch 2.9.1, torchvision 0.24.1, ai-edge-litert 2.0.3, torchao 0.15.0); nightly
  packages were removed. The previous pin `ai-edge-litert-nightly==2.2.0.dev20260202` is no
  longer available on PyPI, so fresh installs had failed.

### Fixed
- Crash on devices with missing system properties (e.g. HiSilicon Kirin).
- The emulator path no longer imports the removed `ai_edge_torch` package.

## Earlier development (2025)

- Emulator-based pipeline (`ab/lite/torch2tflite-all.py`) with Android Virtual Device
  management, the NN-Lite Android app for CPU/GPU/NNAPI inference, crash recovery and device
  analytics.
