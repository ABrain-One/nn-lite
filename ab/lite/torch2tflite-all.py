import sys
import torch
import argparse
from pathlib import Path
import importlib
import ai_edge_torch
from ab.nn.api import data

# --- NHWC wrapper ---
class NHWCWrapper(torch.nn.Module):
    """
    Wraps a PyTorch model to handle NHWC to NCHW format conversion for TFLite.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        # NHWC -> NCHW
        x = x.permute(0, 3, 1, 2).contiguous()
        return self.model(x)

def get_model_configs(model_name: str, num_files: int = 10):
    """Queries the database for a specific model and returns the fastest configurations."""
    df = data(f"nn == '{model_name}'")
    if df.empty:
        raise ValueError(f"No entries found for model '{model_name}'")

    df_sorted = df.sort_values("duration")
    return df_sorted.head(num_files)

def convert_and_export(row, num_classes: int, output_dir: Path):
    """
    Instantiates, wraps, and exports a single model configuration to a TFLite file.
    """
    prm = row["prm"]
    model_name_from_db = row["nn"]
    idx = row.name  # Get the unique index from the DataFrame row

    try:
        size = int(str(prm.get("transform", "")).split("_")[-1])
    except Exception:
        size = 224

    batch = int(prm.get("batch", 1))
    in_shape = (batch, 3, size, size)
    out_shape = (batch, num_classes)
    device = torch.device("cpu")

    # Dynamically import the model module and class
    sys.path.insert(0, ".") # Add current directory to path
    try:
        module = importlib.import_module(f"ab.nn.nn.{model_name_from_db}")
        Net = getattr(module, "Net")
    except Exception as e:
        print(f"[ERROR] Could not import model '{model_name_from_db}': {e}")
        return

    try:
        # Instantiate model (passing correct arguments from the database)
        model_instance = Net(in_shape, out_shape, prm, device)
        model_instance.eval()
        model = NHWCWrapper(model_instance)
        model.eval()
    except Exception as e:
        print(f"[ERROR] Failed to instantiate model for config {idx}: {e}")
        return

    # Create dummy NHWC input
    sample = torch.randn(batch, size, size, 3, dtype=torch.float32)

    # Convert and export using a unique filename
    output_file = output_dir / f"{model_name_from_db}_{idx}.tflite"
    try:
        edge_model = ai_edge_torch.convert(model, (sample,))
        edge_model.export(str(output_file))
        print(f"[SUCCESS] TFLite saved -> {output_file}")
    except Exception as e:
        print(f"[ERROR] Export failed for config {idx}: {e}")

def main():
    """Main execution pipeline."""
    # === CLI ARGUMENTS ===
    parser = argparse.ArgumentParser(description="Convert a PyTorch model from nn-dataset to TFLite (Android-ready NHWC).")
    parser.add_argument("model", type=str, help="Model name from nn-dataset (e.g., AlexNet)")
    
    args = parser.parse_args()

    model_name = args.model
    # The number of classes is hardcoded since it cannot be determined from the database.
    num_classes = 1000 
    output_dir = Path(f'./exports/{model_name}')
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        print(f"Searching for up to 10 configurations for model '{model_name}'...")
        model_configs = get_model_configs(model_name)
        for _, row in model_configs.iterrows():
            convert_and_export(row, num_classes, output_dir)
        print(f"Generated {len(model_configs)} TFLite files.")

    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()