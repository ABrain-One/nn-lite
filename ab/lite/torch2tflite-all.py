import os
import sys
import torch
import argparse
import json
from pathlib import Path
import importlib
import ai_edge_torch

# === CLI ARGUMENTS ===
parser = argparse.ArgumentParser(description="Convert a PyTorch model from nn-dataset to TFLite for each JSON config entry")
parser.add_argument("--model", type=str, required=True, help="Model name from nn-dataset (e.g., AlexNet)")
parser.add_argument("--nn-dataset-path", type=str, required=True, help="Path to nn-dataset repository root")
parser.add_argument("--config-json", type=str, required=True, help="Path to the JSON array file with training configs")
parser.add_argument("--num-classes", type=int, default=100, help="Number of output classes")
parser.add_argument("--output", type=str, required=True, help="Output file path (used as prefix for each accuracy)")
args = parser.parse_args()

# === LOAD CONFIG JSON ARRAY ===
cfg_path = Path(args.config_json)
if not cfg_path.exists():
    print(f"[ERROR] Config JSON not found: {cfg_path}")
    sys.exit(1)
with open(cfg_path, 'r') as f:
    configs = json.load(f)
if not isinstance(configs, list) or not configs:
    print(f"[ERROR] Expected a non-empty list in {cfg_path}")
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

# Iterate and export per config
for cfg in configs:
    acc = cfg.get('accuracy', 0)
    batch = cfg.get('batch', 1)
    dropout = cfg.get('dropout', 0.0)
    lr = cfg.get('lr', 0.0)
    momentum = cfg.get('momentum', 0.0)
    transform = cfg.get('transform', '')
    try:
        size = int(transform.split('_')[-1])
    except:
        size = 224
    in_shape = (batch, 3, size, size)
    out_shape = (batch, args.num_classes)
    prm = {'dropout': dropout, 'lr': lr, 'momentum': momentum, 'transform': transform}
    device = torch.device("cpu")

    try:
        model = Net(in_shape, out_shape, prm, device)
        model.eval()
        print(f"[{acc:.4f}] Instantiated {args.model} with in_shape={in_shape}")
    except Exception as e:
        print(f"[ERROR] Failed instantiate for acc {acc}: {e}")
        continue

    try:
        sample = torch.randn(*in_shape)
        edge_model = ai_edge_torch.convert(model, (sample,))
        file_name = f"{base}_{acc:.4f}.tflite"
        save_path = out_dir / file_name
        print(f"Exporting acc {acc:.4f} → {save_path}")
        edge_model.export(str(save_path))
        print(f"[SUCCESS] {save_path}")
    except Exception as e:
        print(f"[ERROR] Export failed for acc {acc}: {e}")
        continue
