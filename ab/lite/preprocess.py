"""Model-specific input preprocessing shared by INT8 calibration and accuracy evaluation.

Every LEMUR model declares a transform (e.g. ``norm_128`` or ``echo_flip``) in
nn-dataset/ab/nn/transform/. Calibration and evaluation must feed the model
exactly the inputs it was trained on, so both load the transform from there
instead of applying one fixed normalisation to all models.
"""
import importlib.util
from pathlib import Path

# CIFAR-10 statistics used by nn-dataset (ab/nn/loader/cifar-10.py).
CIFAR10_NORM = ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))


def load_transform(transforms_dir, name, norm=CIFAR10_NORM):
    """Return the torchvision transform ``name`` defined in nn-dataset."""
    path = Path(transforms_dir) / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"nn_lite_transform_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.transform(norm)


def preprocess(images, transform, size):
    """Apply ``transform`` to PIL images and return a float32 NCHW array of the given size."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    batch = torch.stack([transform(img) for img in images])
    if batch.shape[-2:] != (size, size):
        batch = F.interpolate(batch, size=(size, size), mode="bilinear", align_corners=False)
    return batch.numpy().astype(np.float32)


def calibration_images(dataset, transform, size, count=50):
    """First ``count`` images of ``dataset`` (PIL image, label pairs), preprocessed for calibration."""
    return preprocess([dataset[i][0] for i in range(count)], transform, size)
