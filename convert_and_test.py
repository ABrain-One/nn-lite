#!/usr/bin/env python3

import os
import shutil
import importlib.util
import inspect

import torch
import numpy as np
import tensorflow as tf

from ab.nn.util.Const import model_script

def main():
    # === Step 1: Copy AlexNet source to temp_model.py ===
    src = model_script("AlexNet")
    dst = "temp_model.py"
    shutil.copy(src, dst)
    print(f"Copied model source from {src} to {dst}\n")

    # === Step 2: Dynamically load temp_model.py ===
    spec = importlib.util.spec_from_file_location("model_mod", dst)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # === Step 3: Inspect & instantiate Net ===
    NetClass = None
    for name, obj in inspect.getmembers(mod, inspect.isclass):
        if name == "Net":
            NetClass = obj
            break
    if NetClass is None:
        raise RuntimeError("Couldn't find class 'Net' in temp_model.py")

    print("Net signature:", inspect.signature(NetClass))

    # Dummy args matching (in_shape, out_shape, prm, device)
    in_shape  = (1, 3, 224, 224)
    out_shape = (1, 1000)
    prm       = {"dropout": 0.5}
    device    = "cpu"

    # Instantiate
    try:
        model = NetClass(in_shape=in_shape, out_shape=out_shape, prm=prm, device=device)
    except TypeError:
        model = NetClass(in_shape, out_shape, prm, device)
    model.eval()
    print("Model instantiated successfully!\n")

    # === Step 4: Convert via the CLI ===
    cmd = (
        "python ab/lite/torch2tflite.py "
        f"--model-script {dst} "
        "--class-name Net "
        "--output ./model.tflite "
        "--input-shape 1,3,224,224"
    )
    print("Running conversion command:\n ", cmd, "\n")
    if os.system(cmd) != 0:
        raise RuntimeError("Error: Conversion to TFLite failed")
    print("Conversion complete: model.tflite created.\n")

    # === Step 5: Dummy inference in PyTorch ===
    dummy = torch.randn(*in_shape)
    with torch.no_grad():
        pt_out = model(dummy).numpy()
    print("PyTorch output (first 5):", pt_out.flatten()[:5], "\n")

    # === Step 6: Dummy inference in TFLite ===
    interpreter = tf.lite.Interpreter(model_path="./model.tflite")
    interpreter.allocate_tensors()
    i = interpreter.get_input_details()[0]["index"]
    o = interpreter.get_output_details()[0]["index"]
    interpreter.set_tensor(i, dummy.numpy().astype(np.float32))
    interpreter.invoke()
    tf_out = interpreter.get_tensor(o)
    print("TFLite output  (first 5):", tf_out.flatten()[:5], "\n")

if __name__ == "__main__":
    main()
