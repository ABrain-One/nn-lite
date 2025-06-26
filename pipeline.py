
import os
import subprocess
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
TFLITE_OUT_DIR = PROJECT_ROOT / 'converted_models'
TFLITE_OUT_DIR.mkdir(exist_ok=True)
TORCH2TFLITE_SCRIPT = PROJECT_ROOT / 'ab' / 'lite' / 'torch2tflite.py'

# === 1. Extract model from nn-dataset ===
def extract_model(model_name: str, output_prefix: Path):
    """
    Use nndataset API to load and save the specified model to disk (no pretrained weights).
    """
    from nndataset import NNDataset
    nd = NNDataset()
    # load_model(name, pretrained=False)
    model = nd.load_model(model_name, pretrained=False)
    pt_path = output_prefix.with_suffix('.pt')
    # save() should serialize to .pt
    model.save(pt_path)
    return pt_path

# === 2. Convert to TFLite ===
def convert_to_tflite(pt_path: Path, tflite_path: Path):
    """
    Convert a PyTorch .pt model to a TensorFlow Lite .tflite file.
    Tries direct import, falls back to CLI invocation.
    """
    try:
        from ab.lite.torch2tflite import torch2tflite
        torch2tflite(
            model_path=str(pt_path),
            tflite_path=str(tflite_path),
            input_size=(1, 3, 224, 224)
        )
    except ImportError:
        subprocess.run([
            'python', str(TORCH2TFLITE_SCRIPT),
            '--model-path', str(pt_path),
            '--tflite-path', str(tflite_path),
            '--input-size', '1,3,224,224'
        ], check=True)
    return tflite_path

# === 3. Deploy to Android device/emulator ===
def deploy_model(tflite_path: Path, device_path: str = '/sdcard/model.tflite') -> str:
    subprocess.run(['adb', 'push', str(tflite_path), device_path], check=True)
    return device_path

# === 4. Trigger Android inference ===
def trigger_inference(activity: str, model_device_path: str):
    cmd = [
        'adb', 'shell', 'am', 'start',
        '-n', activity,
        '--es', 'MODEL_PATH', model_device_path
    ]
    subprocess.run(cmd, check=True)

# === 5. Retrieve and parse results ===
def retrieve_results(device_feedback_path: str = '/sdcard/model_feedback.txt',
                     local_out: Path = PROJECT_ROOT / 'feedback.txt'):
    subprocess.run(['adb', 'pull', device_feedback_path, str(local_out)], check=True)
    success = None
    latency = None
    with open(local_out, 'r') as f:
        for line in f:
            if 'success:' in line.lower():
                success = line.strip().split(':', 1)[1].strip()
            if 'latency:' in line.lower():
                latency = line.strip().split(':', 1)[1].strip()
    return success, latency



# === Main Pipeline ===
def main():
    parser = argparse.ArgumentParser(description='Run automated model verification pipeline')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name to test (e.g. AlexNet). Use "all" to iterate via nn-gpt if available.')
    parser.add_argument('--activity', type=str, required=True,
                        help='Android Activity to start, e.g. com.example.app/.MainActivity')
    args = parser.parse_args()

    model_names = [args.model]
    if args.model.lower() == 'all' and USE_NN_GPT:
        model_names = get_available_models()

    for name in model_names:
        print(f"\n=== Testing model: {name} ===")
        prefix = TFLITE_OUT_DIR / name
        pt_file = extract_model(name, prefix)
        tflite_file = convert_to_tflite(pt_file, prefix.with_suffix('.tflite'))
        device_path = deploy_model(tflite_file)
        trigger_inference(args.activity, device_path)
        success, latency = retrieve_results()
        print(f"Model: {name}, Success: {success}, Latency: {latency}")

if __name__ == '__main__':
    main()
