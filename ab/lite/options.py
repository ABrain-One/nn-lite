"""The command-line options of nn-lite-bench, and the options it restarts itself with.

Kept apart from the pipeline so that both can be tested without PyTorch or a phone.
"""
import argparse
import os

RESTART_EVERY = 50  # models benchmarked before the process restarts to release memory


def _count(text):
    """A whole number of 0 or more."""
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be 0 or more, not {value}")
    return value


def _positive(text):
    """A whole number of 1 or more."""
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be 1 or more, not {value}")
    return value


def parser():
    """The argument parser of nn-lite-bench."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--android-runs", type=_positive, default=20, metavar="N",
                    help="Timed runs of each model on each backend (default: 20)")
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
    ap.add_argument("--serial", default=os.environ.get("ANDROID_SERIAL"),
                    help="Serial number of the phone to benchmark, as listed by 'adb devices' "
                         "(default: $ANDROID_SERIAL, or the only phone connected)")
    ap.add_argument("--restart-every", type=_count, default=RESTART_EVERY, metavar="N",
                    help="Restart the process after every N models to release the memory held by the "
                         f"converter; 0 never restarts (default: {RESTART_EVERY})")
    return ap


def restart_args(args, results_root, serial, checkout=None):
    """The options for the process that continues the run after a restart.

    The run goes on with the same options, the same results folder and the same phone.
    ``--force`` and ``--reinstall-bench`` are not repeated: the new process would otherwise
    delete the progress made so far, or copy the benchmark binary again.
    """
    out = ["--android-runs", str(args.android_runs), "--out", str(results_root),
           "--serial", serial, "--restart-every", str(args.restart_every)]
    if checkout:
        out += ["--dataset-root", str(checkout)]
    if args.model_path:
        out += ["--model-path", *args.model_path, "--input-size", str(args.input_size)]
        if args.calib_dir:
            out += ["--calib-dir", args.calib_dir]
    if args.models:
        out += ["--models", *args.models]
    return out
