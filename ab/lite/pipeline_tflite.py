#!/usr/bin/env python3
import argparse
import os
import sys
import torch

# ① Locate the nn-dataset clone next to nn-lite
script_dir = os.path.dirname(__file__)  # .../nn-lite/ab/lite
repo_root  = os.path.abspath(os.path.join(script_dir, "../../"))  # .../nn-lite
nn_ds_path = os.path.abspath(os.path.join(repo_root, "../nn-dataset"))  # .../nn-dataset

if not os.path.isdir(nn_ds_path):
    raise FileNotFoundError(f"Could not find nn-dataset at {nn_ds_path}")

# ② Prepend it to sys.path so we can import its package
sys.path.insert(0, nn_ds_path)

# ③ Now import the models package
try:
    import nn_dataset.models as nd_models
except ImportError:
    import nndataset.models as nd_models  # fallback if package name differs

# ④ Your conversion helper
from ab.lite.torch2tflite import convert_to_tflite

def load_nn_dataset_model(name: str, pretrained: bool):
    print(f"[1/3] Loading {name} from nn-dataset (pretrained={pretrained})…")
    try:
        ModelCls = getattr(nd_models, name)
    except AttributeError:
        raise ValueError(f"Model '{name}' not found in nn-dataset.models")
    model = ModelCls(pretrained=pretrained)
    model.eval()
    return model

def run_conversion(model: torch.nn.Module, shape: tuple, out_path: str):
    print(f"[2/3] Creating dummy input of shape {shape}…")
    dummy = torch.randn(*shape)

    print(f"[3/3] Converting to TFLite → {out_path}…")
    convert_to_tflite(model=model, input_tensor=dummy, out_path=out_path)
    print(f"\n✅ Done! TFLite file ready at: {os.path.abspath(out_path)}")

def main():
    p = argparse.ArgumentParser(
        description="nn-dataset → PyTorch → TFLite (up to .tflite only)"
    )
    p.add_argument("-m","--model", default="AlexNet",
                   help="nn-dataset model name (e.g. AlexNet, ResNet18)")
    p.add_argument("--pretrained", action="store_true",
                   help="Use pretrained weights (default: random init)")
    p.add_argument("-s","--shape", nargs=4, type=int,
                   default=[1,3,224,224], help="Input tensor shape [N C H W]")
    p.add_argument("-o","--out", default="build/model.tflite",
                   help="Output .tflite file path")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    model = load_nn_dataset_model(args.model, args.pretrained)
    run_conversion(model, tuple(args.shape), args.out)

if __name__ == "__main__":
    main()
