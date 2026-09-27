# Mobile-Ready AI: Verification and Deployment on Edge Devices

<img src='https://abrain.one/img/nnlite-logo.png' width='25%'/>

The original open-source version of the <a href='https://github.com/ABrain-One/NN-Lite/'>NN Lite</a> was developed by <strong>Faraz Kayani</strong>, <strong>Saif U Din</strong> and <strong>Muhammad Ahsan Hussain</strong> at the Computer Vision Laboratory, University of Würzburg, Germany, under the supervision and technical guidance of <strong>Dr. Dmitry Ignatov</strong>, whose foundational work established the basis for the project.

NN-Lite measures how fast PyTorch models run on real Android phones. For every model in the
[LEMUR / NN Dataset](https://github.com/ABrain-One/nn-dataset) it:

1. rebuilds the network and loads its trained weights,
2. converts it to LiteRT (TensorFlow Lite) in **FP32** and full-integer **INT8** (calibrated on real
   CIFAR-10 training images, prepared with the model's own input transform),
3. copies it to a phone over USB and times it with the official `benchmark_model` tool on the **CPU**, **GPU** and **NPU (NNAPI)**,
4. saves one JSON record per model, precision and device back into the dataset.

Runs are unattended and resumable: NN-Lite waits if the USB cable is unplugged, lets the phone
cool down between models, restarts itself every 50 models, and records failures instead of
skipping them. An optional emulator path (Android Studio) is also included.

## Requirements

- Linux (tested on Ubuntu) with Python 3.10 or newer
- `adb` (Android platform tools): `sudo apt install adb`, or the [SDK platform tools](https://developer.android.com/tools/releases/platform-tools)
- An Android phone with USB debugging enabled (see [Connect a phone](#connect-a-phone)); no root is needed
- An internet connection (model weights are downloaded from Hugging Face; the CIFAR-10 training
  set used for INT8 calibration is downloaded once into `nn-dataset/_work/data`)

## Installation

Create and activate a virtual environment (recommended).

For Linux/Mac:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```
For Windows:
```bash
python3 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

Install NN-Lite from PyPI:
```bash
pip install nn-lite --extra-index-url https://download.pytorch.org/whl/cu126
```

Or install it from source:
```bash
git clone https://github.com/ABrain-One/nn-lite.git
cd nn-lite
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu126
```

NN-Lite reads model definitions from, and writes results into, a checkout of the NN Dataset.
By default it looks for an `nn-dataset` folder next to `nn-lite`:
```bash
cd ..
git clone https://github.com/ABrain-One/nn-dataset.git
```
```
your-folder/
├── nn-lite/
└── nn-dataset/
```
To use a checkout elsewhere, pass `--dataset-root /path/to/nn-dataset` or set the
`NN_DATASET_ROOT` environment variable.

## Connect a phone

1. On the phone, open **Settings → About phone** and tap **Build number** seven times to enable developer options.
2. Open **Settings → Developer options** (on some phones under **System**) and turn on **USB debugging**.
3. Connect the phone to the computer with a USB cable and accept the **Allow USB debugging?** prompt on the phone.
4. Check that the phone is visible:
   ```bash
   adb devices
   ```
   It should be listed with the state `device` (not `unauthorized`). Connect only one phone at a time.

Keep the phone charging during long runs. NN-Lite keeps the screen awake and copies the
`benchmark_model` binary to `/data/local/tmp` on the phone automatically.

## Quick start

Benchmark a single model (a few minutes):
```bash
nn-lite-bench --models AirNet
```
From a source checkout, the same command is `python -m ab.lite.torch2tflite --models AirNet`.

NN-Lite converts `AirNet` to FP32 and INT8, times each file on the CPU, GPU and NPU with
20 runs per backend, and writes:
```
nn-dataset/ab/nn/stat/run/tflite/fp32/img-classification_cifar-10_acc_AirNet/android_<device>.json
nn-dataset/ab/nn/stat/run/tflite/int8/img-classification_cifar-10_acc_AirNet/android_<device>.json
```
An abridged record (latencies are in nanoseconds; `unit` is the fastest backend):
```json
{
  "model_name": "AirNet",
  "device_type": "STK-L21",
  "os_version": "10 | HUAWEISTK-L21",
  "valid": true,
  "emulator": false,
  "iterations": 20,
  "duration": 55200000,
  "unit": "GPU",
  "cpu_duration": 329480000, "cpu_min_duration": 310111000, "cpu_max_duration": 344513000, "cpu_std_dev": 9657000.0,
  "gpu_duration": 55200000,  "gpu_min_duration": 53553000,  "gpu_max_duration": 60863000,  "gpu_std_dev": 2064000.0,
  "npu_duration": 364514000, "npu_min_duration": 357856000, "npu_max_duration": 371171000, "npu_std_dev": 6657000.0,
  "total_ram_kb": 3775716, "free_ram_kb": 189496, "available_ram_kb": 1617764, "cached_kb": 1638264,
  "in_dim_0": 1, "in_dim_1": 128, "in_dim_2": 128, "in_dim_3": 3,
  "device_analytics": { "...": "CPU cores, SoC and ARM architecture of the phone" }
}
```
If a backend fails, its error message is stored in `cpu_error`, `gpu_error` or `npu_error`; if
all three fail, the record is kept with `"valid": false`. The full output of every failure is
appended to `nn-dataset/_work/benchmark_errors_<device>.log`.

Benchmark every model (runs for hours; safe to stop and restart at any time):
```bash
nn-lite-bench
```

| Option | Meaning |
|---|---|
| `--models NAME [NAME ...]` | Only process these models |
| `--android-runs N` | Timed runs per backend (default 20) |
| `--dataset-root PATH` | Location of the `nn-dataset` checkout |
| `--force` | Forget the connected phone's progress and start from the beginning |
| `--reinstall-bench` | Copy `benchmark_model` to the phone again |

Progress is stored per phone in `nn-dataset/_work/processing_state_<device>.json`; models
listed there as processed or failed are skipped when the same phone is benchmarked again, while
a new phone starts from the beginning. `--force` resets the progress of the connected phone only.

## Optional: emulator path (Android Studio)

The earlier version of NN-Lite runs models inside an Android emulator through the Android app in
`App/`. It needs the NN Dataset Python package:
```bash
rm -rf db
pip install --no-cache-dir git+https://github.com/ABrain-One/nn-dataset --upgrade --force --extra-index-url https://download.pytorch.org/whl/cu126
```

Install Android Studio 'Android Studio Narwhal 3 Feature Drop | 2025.1.3' (outside of the virtual environment) with the ready-made script (Linux):
```bash
chmod +x install-android-studio.sh
./install-android-studio.sh
```

Or install it manually from the [Android Studio archive](https://developer.android.com/studio/archive):
```bash
sudo apt update
sudo apt install openjdk-17-jdk
cd ~/Downloads
unzip android-studio-*.zip
sudo mv android-studio /opt/
/opt/android-studio/bin/studio.sh
```
In Android Studio, select `App` and import it as a project, then go to
**Tools → Device Manager** and add a new device with the **+** symbol (e.g. Pixel 5).

Set up the Android SDK environment variables by adding these lines to the end of `~/.bashrc`,
using your own paths (shown under **Tools → Device Manager → Android SDK Location**):
```bash
export ANDROID_SDK_ROOT="$HOME/Android/Sdk"
export ANDROID_HOME="$HOME/Android/Sdk"
export PATH="$PATH:$HOME/Android/Sdk/cmdline-tools/latest/bin:$HOME/.local/bin"
```

Run all models, a single model, or several models:
```bash
python -m ab.lite.torch2tflite-all
python -m ab.lite.torch2tflite-all AirNet
python -m ab.lite.torch2tflite-all AirNet ga-196 ga-197 ga-198
```

## Running the tests

The unit tests cover output parsing, error extraction and the result schema. They need neither a
phone nor PyTorch or TensorFlow:
```bash
pip install pytest
python -m pytest tests
```

## Contributing and support

Bug reports, questions and feature requests are welcome in the
[issue tracker](https://github.com/ABrain-One/nn-lite/issues). See [CONTRIBUTING.md](CONTRIBUTING.md)
for how to propose changes and how the project is maintained.

## Citation

If you find this project to be useful for your research, please consider citing our articles:
```bibtex
@article{ABrain.NN-Lite,
    title = {AI on the Edge: An Automated Pipeline for PyTorch-to-Android Deployment and Benchmarking},
	author = {Saif U Din and Muhammad Ahsan Hussain and Mohsin Ikram and Faraz Kayani and Dmitry Ignatov and Radu Timofte},
	doi = {10.20944/preprints202511.1831.v2},
	url = {https://doi.org/10.20944/preprints202511.1831.v2},
	year = 2026,
	month = {July},
	publisher = {Preprints},
	journal = {Preprints}
}

@InProceedings{ABrain.MobileDenoising,
	title = {Real Image Denoising with Knowledge Distillation for High-Performance Mobile {NPUs}},
	author = {Faraz Kayani and Sarmad Kayani and Asad Ahmed and Radu Timofte and Dmitry Ignatov},
	booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},	
	pages = {3792--3800},		
	year={2026}
}

@InProceedings{ABrain.MobileAgeNet,
	title = {{MobileAgeNet}: Lightweight Facial Age Estimation for Mobile Deployment},
	author = {Arun Kumar and Aswathy Baiju and Radu Timofte and Dmitry Ignatov},
	booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},	
	pages = {3810--3818},		
	year={2026}
}

```

## License

NN-Lite is released under the [MIT License](LICENSE).

#### The idea and leadership of Dr. Ignatov
