import os
import sys
import torch
import argparse
import json
from pathlib import Path
import importlib
import ai_edge_torch

# === CLI ARGUMENTS ===
parser = argparse.ArgumentParser(description="Convert a PyTorch model from nn-dataset to TFLite (Android-ready NHWC) for each JSON config entry")
parser.add_argument("--model", type=str, required=True, help="Model name from nn-dataset (e.g., AlexNet)")
parser.add_argument("--nn-dataset-path", type=str, required=True, help="Path to nn-dataset repository root")
parser.add_argument("--config-json", type=str, required=True, help="Path to the JSON file or JSON array file with training configs")
parser.add_argument("--num-classes", type=int, default=100, help="Number of output classes")
parser.add_argument("--output", type=str, required=True, help="Output file path (used as prefix for each accuracy)")
args = parser.parse_args()

# === LOAD CONFIG JSON (accepts list or single object) ===
cfg_path = Path(args.config_json)
if not cfg_path.exists():
    print(f"[ERROR] Config JSON not found: {cfg_path}")
    sys.exit(1)
with open(cfg_path, 'r') as f:
    loaded = json.load(f)

# Allow either a JSON array or a single JSON object
if isinstance(loaded, dict):
    configs = [loaded]
elif isinstance(loaded, list) and loaded:
    configs = loaded
else:
    print(f"[ERROR] Expected a non-empty list or a single object in {cfg_path}")
    sys.exit(1)

# Prepare output directory and base name
out_path = Path(args.output)
out_dir = out_path.parent
base = out_path.stem  # filename without extension
out_dir.mkdir(parents=True, exist_ok=True)

# Setup model import
sys.path.insert(0, args.nn_dataset_path)
try:
    model_module = importlib.import_module(f"ab.nn.nn.{args.model}")
    Net = getattr(model_module, "Net")
except Exception as e:
    print(f"[ERROR] Could not import model '{args.model}': {e}")
    sys.exit(1)

# === Wrapper to accept NHWC and feed NCHW to PyTorch models ===
class NHWCWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        # Input x expected NHWC from Android/TFLite
        # Convert to NCHW for PyTorch backbone
        x = x.permute(0, 3, 1, 2).contiguous()
        return self.model(x)

# Iterate and export per config
for cfg in configs:
    acc = cfg.get('accuracy', 0)
    batch = int(cfg.get('batch', 1))
    dropout = float(cfg.get('dropout', 0.0))
    lr = float(cfg.get('lr', 0.0))
    momentum = float(cfg.get('momentum', 0.0))
    transform = cfg.get('transform', '') or ''
    try:
        size = int(str(transform).split('_')[-1])
    except Exception:
        size = 224

    # PyTorch model (backbone) remains NCHW internally
    nchw_in_shape = (batch, 3, size, size)
    out_shape = (batch, args.num_classes)
    prm = {'dropout': dropout, 'lr': lr, 'momentum': momentum, 'transform': transform}
    device = torch.device("cpu")

    try:
        backbone = Net(nchw_in_shape, out_shape, prm, device)
        backbone.eval()
        model = NHWCWrapper(backbone)  # Accept NHWC externally
        model.eval()
        print(f"[{acc:.4f}] Instantiated {args.model} with NHWC external input (backbone NCHW) size={size}")
    except Exception as e:
        print(f"[ERROR] Failed instantiate for acc {acc}: {e}")
        continue

    try:
        # Sample is NHWC (Android/TFLite friendly)
        sample = torch.randn(batch, size, size, 3, dtype=torch.float32)

        # Convert via ai_edge_torch using NHWC sample so exported TFLite takes NHWC
        edge_model = ai_edge_torch.convert(model, (sample,))

        file_name = f"{base}_{acc:.4f}.tflite"
        save_path = out_dir / file_name
        print(f"Exporting acc {acc:.4f} → {save_path}")
        edge_model.export(str(save_path))
        print(f"[SUCCESS] {save_path}")
    except Exception as e:
        print(f"[ERROR] Export failed for acc {acc}: {e}")
        continue
