"""Models given with --model-path, benchmarked without the NN Dataset.

Each model is given as a file, or as a folder containing such files:

* ``name.pt2``: a model saved with ``torch.export.save``. It needs no Python code, and
  its input shape is stored in the file.
* ``name.py`` with ``name.pt`` or ``name.pth`` next to it: the model's code, and either
  its weights (``torch.save(model.state_dict(), ...)``) or the whole model
  (``torch.save(model, ...)``).

A ``.pt`` file on its own cannot be converted: it holds the weights, or references to
the model's classes, but not the code PyTorch needs to rebuild the network.

The ``.py`` file builds the network with ``create_model()`` if it defines one, otherwise
with ``Net()`` or its only model class. It may define ``input_transform``, a callable that
turns a PIL image into a CHW tensor, for INT8 calibration.
"""
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

WEIGHT_SUFFIXES = (".pt", ".pth")
MODEL_SUFFIXES = (".pt2", ".py") + WEIGHT_SUFFIXES
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
IMAGENET_NORM = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

_loaded = set()      # names of the model modules registered in sys.modules by _load_code
_main_bound = set()  # names of the model classes set in __main__ by _load_code


@dataclass(frozen=True)
class LocalModel:
    name: str
    exported: Path = None  # .pt2 file saved with torch.export
    code: Path = None      # .py file defining the network
    weights: Path = None   # .pt or .pth file next to it


def _weights_for(code):
    return next((code.with_suffix(s) for s in WEIGHT_SUFFIXES if code.with_suffix(s).exists()), None)


def _model_for(path):
    """The LocalModel for one file, or a ValueError explaining why it cannot be used."""
    if path.suffix == ".pt2":
        return LocalModel(path.stem, exported=path)
    if path.suffix == ".py":
        weights = _weights_for(path)
        if weights is None:
            raise ValueError(f"{path}: no {path.stem}.pt or {path.stem}.pth next to it")
        return LocalModel(path.stem, code=path, weights=weights)
    if path.suffix in WEIGHT_SUFFIXES:
        code = path.with_suffix(".py")
        if not code.exists():
            raise ValueError(f"{path}: no {path.stem}.py next to it. A {path.suffix} file does not contain "
                             "the network's code, so it cannot be converted on its own; put the model's "
                             ".py file next to it, or save the model with torch.export.save as a .pt2 file")
        return LocalModel(path.stem, code=code, weights=path)
    raise ValueError(f"{path}: not a .pt2, .py, .pt or .pth file")


def find_models(paths):
    """Collect the models in ``paths`` (files or folders).

    Returns ``({name: LocalModel}, warnings)``. Problems with a file given directly are
    errors; files in a folder that cannot be used are reported as warnings, and ``.py``
    files without weights (e.g. helper modules) are ignored.
    """
    models, warnings = {}, []

    def add(model):
        if models.get(model.name, model) != model:
            raise ValueError(f"two different models are called {model.name}")
        models[model.name] = model

    for path in map(Path, paths):
        if path.is_dir():
            for f in sorted(path.iterdir()):
                if f.suffix not in MODEL_SUFFIXES or (f.suffix == ".py" and _weights_for(f) is None):
                    continue
                try:
                    add(_model_for(f))
                except ValueError as e:
                    warnings.append(str(e))
        elif path.exists():
            add(_model_for(path))
        else:
            raise ValueError(f"{path} does not exist")
    if not models:
        raise ValueError("no models found in " + ", ".join(map(str, paths)))
    return models, warnings


def _load_code(path):
    """Import the model's .py file.

    Its model classes are also made available in ``__main__``, where ``torch.save`` records
    them when the model was saved from a script that was run directly. They replace the
    classes of a previously loaded model, as scripts often use the same class names.
    """
    import torch

    if str(path.parent) not in sys.path:
        sys.path.append(str(path.parent))  # for helper modules next to the model's file
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    # Register the module under its own name, as torch.save records classes by module name,
    # but never replace a module that was not loaded from a model file (e.g. "torch").
    if path.stem not in sys.modules or path.stem in _loaded:
        sys.modules[path.stem] = module
        _loaded.add(path.stem)
    spec.loader.exec_module(module)
    main = sys.modules["__main__"]
    for name, obj in vars(module).items():
        if (isinstance(obj, type) and issubclass(obj, torch.nn.Module)
                and (not hasattr(main, name) or name in _main_bound)):
            setattr(main, name, obj)
            _main_bound.add(name)
    return module


def _build(module):
    """Create the network defined by ``module``."""
    import torch

    if hasattr(module, "create_model"):
        return module.create_model()
    own = [obj for obj in vars(module).values()
           if isinstance(obj, type) and issubclass(obj, torch.nn.Module) and obj.__module__ == module.__name__]
    cls = getattr(module, "Net", None) or (own[0] if len(own) == 1 else None)
    if cls is None:
        raise ValueError(f"{module.__file__}: define create_model(), a class called Net, or only one model class")
    try:
        return cls()
    except TypeError as e:
        raise ValueError(f"{cls.__name__}() needs arguments ({e}); define create_model() in "
                         f"{module.__file__} to build the network") from e


def _state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if isinstance(checkpoint.get(key), dict):
                return checkpoint[key]
    return checkpoint


def load_local_model(model, input_size=224):
    """Return ``(network in eval mode, NCHW input shape, input transform or None)``."""
    import torch

    if model.exported:
        program = torch.export.load(str(model.exported))
        inputs = program.example_inputs[0]
        if len(inputs) != 1 or inputs[0].dim() != 4:
            raise ValueError(f"{model.exported}: only models with one 4-D (NCHW) image input are supported")
        network = program.module()
        # The graph was fixed when the model was exported. Exported modules do not support
        # .eval(), so only the flags are cleared, which the converter would otherwise report
        # as "converted in training mode".
        for m in network.modules():
            m.training = False
        return network, tuple(inputs[0].shape), None

    module = _load_code(model.code)
    try:
        checkpoint = torch.load(model.weights, map_location="cpu", weights_only=True)
    except Exception:
        # A whole model saved with torch.save(model). Loading it can run code stored in
        # the file, which is why PyTorch refuses it by default; it is done here only for
        # the files the user gives with --model-path.
        checkpoint = torch.load(model.weights, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model"), torch.nn.Module):
        checkpoint = checkpoint["model"]
    if isinstance(checkpoint, torch.nn.Module):
        network = checkpoint
    else:
        network = _build(module)
        network.load_state_dict(_state_dict(checkpoint))
    return network.eval(), (1, 3, input_size, input_size), getattr(module, "input_transform", None)


def default_transform(shape):
    """Resize to the model's input and normalise with the ImageNet statistics."""
    import torchvision.transforms as T

    return T.Compose([T.Resize(tuple(shape[2:])), T.ToTensor(), T.Normalize(*IMAGENET_NORM)])


def calibration_set(folder, count=50):
    """Up to ``count`` images from ``folder`` as (PIL image, label) pairs for INT8 calibration."""
    from PIL import Image

    files = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)[:count]
    if not files:
        raise ValueError(f"--calib-dir {folder}: no images ({', '.join(IMAGE_SUFFIXES)}) found")
    return [(Image.open(p).convert("RGB"), 0) for p in files]
