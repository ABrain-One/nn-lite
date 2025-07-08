import os
import sys
import torch
import argparse
from pathlib import Path
import importlib
import ai_edge_torch

# If we run pipeline with nn-dataset models path it is providing the .tflite file which is executable to mobile. python torch2tflite.py \  --model AlexNet \  --nn-dataset-path /home/saif/lab/nn-dataset \  --output ./tflites/alexnet.tflite


# === CLI ARGUMENTS ===
parser = argparse.ArgumentParser(description="Convert a PyTorch model from nn-dataset to TFLite format")
parser.add_argument("--model", type=str, required=True, help="Model name from nn-dataset (e.g., AlexNet, ResNet18)")
parser.add_argument("--output", type=str, default="model.tflite", help="Output path for .tflite file")
parser.add_argument("--nn-dataset-path", type=str, required=True, help="Path to nn-dataset repository")
args = parser.parse_args()

# === ADD nn-dataset TO PATH ===
sys.path.insert(0, args.nn_dataset_path)

# === IMPORT MODEL ===
try:
    model_module = importlib.import_module(f"ab.nn.nn.{args.model}")
    Net = getattr(model_module, "Net")
except (ImportError, AttributeError) as e:
    print(f"[ERROR] Could not import model '{args.model}' from nn-dataset: {e}")
    sys.exit(1)

# === INSTANTIATE MODEL ===
in_shape = (1, 3, 224, 224)
out_shape = (1000,)
prm = {
    'dropout': False,
    'batch_norm': False,
    'activation': 'relu'
}
device = torch.device("cpu")

try:
    model = Net(in_shape, out_shape, prm, device)
    model.eval()
except Exception as e:
    print(f"[ERROR] Failed to instantiate model '{args.model}': {e}")
    sys.exit(1)

# === CONVERT TO TFLITE ===
try:
    sample_input = torch.randn(*in_shape)
    edge_model = ai_edge_torch.convert(model, (sample_input,))

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Exporting to {output_path}")

    edge_model.export(str(output_path))
    print(f"[SUCCESS] TFLite model exported to {output_path}")
except Exception as e:
    print(f"[ERROR] TFLite export failed: {e}")
    sys.exit(1)
