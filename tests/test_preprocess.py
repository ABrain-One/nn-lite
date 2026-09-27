"""Tests for model-specific preprocessing used by INT8 calibration."""
import pytest

from ab.lite.preprocess import CIFAR10_NORM, load_transform


def test_load_transform_passes_cifar10_statistics(tmp_path):
    (tmp_path / "fake.py").write_text("def transform(norm):\n    return ('built with', norm)\n")
    assert load_transform(tmp_path, "fake") == ("built with", CIFAR10_NORM)


def test_cifar10_statistics_match_nn_dataset():
    assert CIFAR10_NORM == ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))


def test_calibration_images_use_model_transform_and_size():
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    from ab.lite.preprocess import calibration_images

    dataset = [(i, 0) for i in range(60)]
    to_tensor = lambda i: torch.full((3, 16, 16), float(i))
    calib = calibration_images(dataset, to_tensor, size=32, count=50)
    assert calib.shape == (50, 3, 32, 32)
    assert calib.dtype == np.float32
    assert calib[7].min() == calib[7].max() == 7.0  # real samples, in dataset order, resized to 32x32


class _Out:
    def __init__(self, side):
        self.shape = (3, side, side)


def test_input_size_comes_from_the_transform_output():
    from ab.lite.preprocess import input_size
    # e.g. CenterCrop(32) then Resize((64, 64)): the source mentions 32 first, but the model sees 64x64
    assert input_size(lambda img: _Out(64), probe=object()) == 64
    assert input_size(lambda img: _Out(128), probe=object()) == 128
