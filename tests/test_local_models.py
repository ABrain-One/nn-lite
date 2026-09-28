"""Tests for benchmarking models from local files (--model-path).

Finding the files runs without PyTorch; loading them needs PyTorch and is skipped
without it.
"""
import subprocess
import sys

import pytest

from ab.lite.local_models import LocalModel, find_models

MODEL = """\
import torch.nn as nn

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3)

    def forward(self, x):
        return self.conv(x)

class Net(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.block = Block()
        self.fc = nn.Linear(8, num_classes)

    def forward(self, x):
        return self.fc(self.block(x).mean((2, 3)))
"""


def touch(path, text=""):
    path.write_text(text)
    return path


# --- finding models ------------------------------------------------------

def test_exported_file(tmp_path):
    pt2 = touch(tmp_path / "resnet.pt2")
    models, warnings = find_models([pt2])
    assert models == {"resnet": LocalModel("resnet", exported=pt2)}
    assert warnings == []


def test_code_and_weights_are_paired_from_either_file(tmp_path):
    code, weights = touch(tmp_path / "net.py"), touch(tmp_path / "net.pth")
    expected = {"net": LocalModel("net", code=code, weights=weights)}
    assert find_models([code])[0] == expected
    assert find_models([weights])[0] == expected
    assert find_models([code, weights])[0] == expected


def test_weights_without_code_are_an_error(tmp_path):
    with pytest.raises(ValueError, match="cannot be converted on its own"):
        find_models([touch(tmp_path / "net.pt")])


def test_code_without_weights_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="no net.pt or net.pth"):
        find_models([touch(tmp_path / "net.py")])


def test_folder(tmp_path):
    touch(tmp_path / "a.pt2")
    touch(tmp_path / "b.py"), touch(tmp_path / "b.pt")
    touch(tmp_path / "helpers.py")  # no weights: a helper module, ignored
    touch(tmp_path / "orphan.pt")   # no code: reported
    touch(tmp_path / "notes.txt")
    models, warnings = find_models([tmp_path])
    assert sorted(models) == ["a", "b"]
    assert len(warnings) == 1 and "orphan.pt" in warnings[0]


def test_missing_path_and_empty_folder(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        find_models([tmp_path / "nothing.pt2"])
    with pytest.raises(ValueError, match="no models found"):
        find_models([tmp_path])


def test_same_name_in_two_folders_is_an_error(tmp_path):
    (tmp_path / "x").mkdir(), (tmp_path / "y").mkdir()
    touch(tmp_path / "x" / "m.pt2"), touch(tmp_path / "y" / "m.pt2")
    with pytest.raises(ValueError, match="two different models are called m"):
        find_models([tmp_path / "x" / "m.pt2", tmp_path / "y" / "m.pt2"])


# --- loading models ------------------------------------------------------

def outputs_match(tmp_path, save, input_size=32):
    """Save a model with ``save(model, folder)`` and check that loading it gives the same outputs."""
    torch = pytest.importorskip("torch")
    from ab.lite.local_models import _load_code, load_local_model

    (tmp_path / "net.py").write_text(MODEL)
    original = _load_code(tmp_path / "net.py").Net().eval()
    save(original, tmp_path)
    models, _ = find_models([tmp_path])
    loaded, shape, transform = load_local_model(models["net"], input_size)
    x = torch.randn(*shape)
    with torch.no_grad():
        assert torch.allclose(loaded(x), original(x))
    return shape


def test_load_weights(tmp_path):
    torch = pytest.importorskip("torch")
    shape = outputs_match(tmp_path, lambda m, d: torch.save(m.state_dict(), d / "net.pth"), input_size=64)
    assert shape == (1, 3, 64, 64)


def test_load_checkpoint_dictionary(tmp_path):
    torch = pytest.importorskip("torch")
    outputs_match(tmp_path, lambda m, d: torch.save({"epoch": 5, "state_dict": m.state_dict()}, d / "net.pt"))


def test_load_whole_model(tmp_path):
    torch = pytest.importorskip("torch")
    outputs_match(tmp_path, lambda m, d: torch.save(m, d / "net.pt"))


def test_load_whole_model_saved_from_a_script(tmp_path):
    """torch.save records classes of a script run directly as __main__.Net."""
    torch = pytest.importorskip("torch")
    from ab.lite.local_models import load_local_model

    (tmp_path / "net.py").write_text(MODEL)
    script = tmp_path / "train.py"
    script.write_text(MODEL + "\nimport torch\ntorch.save(Net(), 'net.pt')\n")
    subprocess.run([sys.executable, str(script)], cwd=tmp_path, check=True)
    loaded, shape, _ = load_local_model(find_models([tmp_path])[0]["net"], 32)
    assert loaded(torch.randn(*shape)).shape == (1, 10)


def test_create_model_and_input_transform(tmp_path):
    torch = pytest.importorskip("torch")
    from ab.lite.local_models import load_local_model

    (tmp_path / "net.py").write_text(MODEL + """
def create_model():
    return Net(num_classes=4)

def input_transform(image):
    return image
""")
    import importlib.util
    spec = importlib.util.spec_from_file_location("net_for_weights", tmp_path / "net.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    torch.save(module.create_model().state_dict(), tmp_path / "net.pth")
    loaded, shape, transform = load_local_model(find_models([tmp_path])[0]["net"], 32)
    assert loaded(torch.randn(*shape)).shape == (1, 4)
    assert transform("image") == "image"


def test_load_exported_program(tmp_path):
    torch = pytest.importorskip("torch")
    from ab.lite.local_models import _load_code, load_local_model

    (tmp_path / "code.py").write_text(MODEL)
    original = _load_code(tmp_path / "code.py").Net().eval()
    x = torch.randn(1, 3, 48, 40)
    torch.export.save(torch.export.export(original, (x,)), tmp_path / "exported.pt2")
    loaded, shape, transform = load_local_model(LocalModel("exported", exported=tmp_path / "exported.pt2"))
    assert shape == (1, 3, 48, 40) and transform is None
    assert not any(m.training for m in loaded.modules())
    with torch.no_grad():
        assert torch.allclose(loaded(x), original(x))


def test_whole_models_from_different_scripts_with_the_same_class_name(tmp_path):
    """Scripts often all call their class Net; each model must load with its own class."""
    torch = pytest.importorskip("torch")
    from ab.lite.local_models import load_local_model

    for name, width in (("first", 4), ("second", 6)):
        code = MODEL.replace("nn.Linear(8, num_classes)", f"nn.Linear(8, {width})")
        (tmp_path / f"{name}.py").write_text(code)
        script = tmp_path / f"save_{name}.py"
        script.write_text(code + f"\nimport torch\ntorch.save(Net(), '{name}.pt')\n")
        subprocess.run([sys.executable, str(script)], cwd=tmp_path, check=True)
        script.unlink()
    models, _ = find_models([tmp_path])
    for name, width in (("first", 4), ("second", 6)):
        loaded, shape, _ = load_local_model(models[name], 32)
        assert loaded(torch.randn(*shape)).shape == (1, width)
