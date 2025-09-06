import os
import sys
import json
from pathlib import Path
from ab.nn.api import check_nn, data
import importlib.util, torch, ai_edge_torch
import inspect

# Force CPU usage to avoid JAX device conflicts
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

tflite_dir = Path('./exports_tflite')
tflite_dir.mkdir(parents=True, exist_ok=True)

# Create directory for JSON files
json_base_dir = Path('../../ab/nn/stat/train')
json_base_dir.mkdir(parents=True, exist_ok=True)


def find_tmp_model_file(model_name: str, prefix: str):
    """Find the generated tmp compiled model file (.pyc) matching model_name and prefix."""
    # Use the project root path that was added to sys.path
    project_root = Path(sys.path[0])
    tmp_dir = project_root / 'out/nn/tmp/__pycache__'
    
    print(f"DEBUG: Looking in directory: {tmp_dir}")
    print(f"DEBUG: Directory exists: {tmp_dir.exists()}")
    
    if not tmp_dir.exists():
        # Try an alternative path
        tmp_dir_alt = Path('./out/nn/tmp/__pycache__')
        print(f"DEBUG: Trying alternative directory: {tmp_dir_alt}")
        print(f"DEBUG: Alternative directory exists: {tmp_dir_alt.exists()}")
        if tmp_dir_alt.exists():
            tmp_dir = tmp_dir_alt
    
    if not tmp_dir.exists():
        print(f"DEBUG: Directory {tmp_dir} does not exist")
        return None
    
    # List all files in the directory
    all_files = list(tmp_dir.iterdir())
    print(f"DEBUG: All files in {tmp_dir}: {all_files}")
    
    # Look for .pyc files that contain the prefix in their name
    for file in all_files:
        if file.suffix == '.pyc' and prefix in file.name:
            print(f"DEBUG: Found match: {file}")
            return file
    
    print(f"DEBUG: No match found for prefix {prefix}")
    return None


def convert_to_serializable(obj):
    """Convert non-serializable objects to JSON-serializable formats."""
    if hasattr(obj, 'item'):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


def train_and_export_tflite(model_name: str, min_files: int = 10):
    # Create model-specific JSON directory
    json_model_dir = json_base_dir / model_name
    json_model_dir.mkdir(parents=True, exist_ok=True)

    df = data(f"nn == '{model_name}'")
    if df.empty:
        raise ValueError(f"No database entries found for model '{model_name}'")

    df_sorted = df.sort_values("duration")
    exported_files = []

    # Find the next available file number
    existing_files = list(json_model_dir.glob("*.json"))
    file_counter = 1
    if existing_files:
        # Extract numbers from existing files and find the max
        numbers = []
        for file in existing_files:
            try:
                numbers.append(int(file.stem))
            except ValueError:
                continue
        if numbers:
            file_counter = max(numbers) + 1

    for idx, row in df_sorted.head(min_files).iterrows():
        prm = row["prm"].copy()
        if "epoch" not in prm:
            prm["epoch"] = 1

        prefix = f"{model_name}_{idx}"

        # Save only duration in JSON format
        duration_data = {"duration": convert_to_serializable(row.get("duration"))}
        
        # Create JSON file with incremental numbering
        json_file = json_model_dir / f"{file_counter}.json"
        with open(json_file, 'w') as f:
            json.dump([duration_data], f, indent=4)
        print(f"[SUCCESS] Saved JSON → {json_file}")
        
        # Increment counter for next file
        file_counter += 1

        try:
            check_nn(
                nn_code=row["nn_code"],
                task=row["task"],
                dataset=row["dataset"],
                metric=row["metric"],
                prm=prm,
                prefix=prefix,
                save_to_db=False,
                export_onnx=False,
                epoch_limit_minutes=5 * 60
            )
        except Exception as e:
            print(f"Warning: Training failed for prefix {prefix}: {e}")
            # Continue to next iteration even if training fails
            continue

        tmp_model_path = find_tmp_model_file(model_name, prefix)
        if not tmp_model_path or not tmp_model_path.exists():
            print(f"Warning: tmp model not found for prefix {prefix}")
            continue

        try:
            spec = importlib.util.spec_from_file_location(tmp_model_path.stem, tmp_model_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            net_cls = getattr(mod, 'Net')

            image_size = 224
            num_classes = 1000
            batch_size = int(prm.get("batch", 1))
            
            in_shape = (batch_size, 3, image_size, image_size)
            out_shape = (batch_size, num_classes)
            device = "cpu"

            arg_map = {
                'in_shape': in_shape,
                'out_shape': out_shape,
                'num_classes': num_classes,
                'prm': prm,
                'device': device
            }

            constructor_params = inspect.signature(net_cls.__init__).parameters
            args_to_pass = []
            for param_name in constructor_params:
                if param_name in arg_map:
                    args_to_pass.append(arg_map[param_name])
                elif param_name != 'self':
                    raise ValueError(f"Required parameter '{param_name}' not found for model constructor.")
            
            model_instance = net_cls(*args_to_pass)
            model_instance.eval()

            # Create input tensor in NCHW format (batch, channels, height, width)
            sample = torch.randn(batch_size, 3, image_size, image_size, device=device)
            
            tflite_model = ai_edge_torch.convert(model_instance, (sample,))
            
            file_name = f"{model_name}_{idx}.tflite"
            output_file = tflite_dir / file_name
            tflite_model.export(str(output_file))
            print(f"[SUCCESS] Exported TFLite → {output_file}")
            exported_files.append(output_file)
        except Exception as e:
            print(f"Error during model conversion for prefix {prefix}: {e}")
            continue

    return exported_files


def main():
    if len(sys.argv) < 2:
        print("Usage: python test.py <model_name>")
        sys.exit(1)

    model_name = sys.argv[1]
    min_files = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    
    try:
        files = train_and_export_tflite(model_name, min_files=min_files)
        print(f"Generated {len(files)} TFLite files in {tflite_dir}")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
