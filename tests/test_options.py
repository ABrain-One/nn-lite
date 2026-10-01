"""Tests for the options that nn-lite-bench restarts itself with.

After every --restart-every models the process restarts and continues the run with the
options built by restart_args. These tests run without PyTorch or a phone.
"""
import pytest

from ab.lite.options import parser, restart_args

# Options that must not be repeated after a restart: --force would delete the progress made
# so far, and --reinstall-bench would copy the benchmark binary to the phone again.
NOT_REPEATED = {"force", "reinstall_bench"}


@pytest.fixture(autouse=True)
def no_android_serial(monkeypatch):
    monkeypatch.delenv("ANDROID_SERIAL", raising=False)


def restarted(args, results_root, serial, checkout=None):
    return parser().parse_args(restart_args(args, results_root, serial, checkout))


def assert_same_run(before, after):
    for option, value in vars(before).items():
        if option in NOT_REPEATED:
            assert getattr(after, option) is False, option
        else:
            assert getattr(after, option) == value, option


def test_every_option_is_handled_by_the_restart():
    """A new option must be added to restart_args, or to NOT_REPEATED if it must not be kept."""
    kept = {"android_runs", "dataset_root", "out", "models", "model_path", "input_size", "calib_dir",
            "serial", "restart_every"}
    options = {a.dest for a in parser()._actions if a.dest != "help"}
    assert options == kept | NOT_REPEATED


def test_restart_keeps_a_dataset_run(tmp_path):
    checkout, out = str(tmp_path / "nn-dataset"), str(tmp_path / "results")
    args = parser().parse_args(["--models", "AirNet", "ResNet", "--android-runs", "7", "--serial", "R58M12ABCDE",
                                "--restart-every", "3", "--dataset-root", checkout, "--out", out,
                                "--force", "--reinstall-bench"])
    assert_same_run(args, restarted(args, out, "R58M12ABCDE", checkout))


def test_restart_keeps_a_model_path_run(tmp_path):
    out = str(tmp_path / "results")
    args = parser().parse_args(["--model_path", "my_models/", "net.pt2", "--input-size", "96", "--calib-dir",
                                "images", "--out", out, "--serial", "192.168.1.20:5555", "--force"])
    assert_same_run(args, restarted(args, out, "192.168.1.20:5555"))


def test_restart_keeps_the_phone_chosen_at_start_up():
    """Without --serial the only connected phone is used; after a restart it must be the same one."""
    args = parser().parse_args([])
    assert args.serial is None
    assert restarted(args, "nn-lite-results", "R58M12ABCDE").serial == "R58M12ABCDE"


def test_restart_every():
    assert parser().parse_args([]).restart_every == 50
    assert parser().parse_args(["--restart-every", "0"]).restart_every == 0  # never restart
    with pytest.raises(SystemExit):
        parser().parse_args(["--restart-every", "-1"])


def test_android_runs():
    assert parser().parse_args([]).android_runs == 20
    for runs in ("0", "-5"):  # benchmark_model would then time no runs at all
        with pytest.raises(SystemExit):
            parser().parse_args(["--android-runs", runs])
