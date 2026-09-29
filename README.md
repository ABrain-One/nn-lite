# Mobile-Ready AI: Verification and Deployment on Edge Devices

<img src='https://abrain.one/img/nnlite-logo.png' width='25%'/>

The original open-source version of the <a href='https://github.com/ABrain-One/NN-Lite/'>NN Lite</a> was developed by <strong>Faraz Kayani</strong>, <strong>Saif U Din</strong> and <strong>Muhammad Ahsan Hussain</strong> at the Computer Vision Laboratory, University of Würzburg, Germany, under the supervision and technical guidance of <strong>Dr. Dmitry Ignatov</strong>, whose foundational work established the basis for the project.

NN-Lite measures how fast PyTorch models run on real Android phones. For every model in the
[LEMUR / NN Dataset](https://github.com/ABrain-One/nn-dataset) it:

1. rebuilds the network and loads its trained weights,
2. converts it to LiteRT (TensorFlow Lite) in **FP32** and full-integer **INT8** (calibrated on real
   CIFAR-10 training images, prepared with the model's own input transform),
3. copies it to a phone over USB and times it with the official `benchmark_model` tool on the **CPU**, **GPU** and **NPU (NNAPI)**,
4. saves one JSON record per model, precision and device, in the same layout as the dataset.

Runs are unattended and resumable: NN-Lite waits if the USB cable is unplugged, lets the phone
cool down between models, restarts itself every 50 models (`--restart-every`), and records
failures instead of skipping them. An optional emulator path (Android Studio) is also included.

## Requirements

- Linux (tested on Ubuntu) or macOS on Apple Silicon, with Python 3.10 or newer
- `adb` (Android platform tools): `sudo apt install adb` on Linux, `brew install --cask android-platform-tools`
  on macOS, or the [SDK platform tools](https://developer.android.com/tools/releases/platform-tools)
- An Android phone with USB debugging enabled (see [Connect a phone](#connect-a-phone)); no root is needed
- An internet connection and about 2 GB of free disk space: the first run downloads the LEMUR
  database (about 1.2 GB unpacked) and the CIFAR-10 training set used for INT8 calibration;
  model weights are downloaded from Hugging Face as they are needed

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
pip install nn-lite
```
This also installs the [NN Dataset](https://github.com/ABrain-One/nn-dataset) package, from
which NN-Lite reads the models, so nothing else needs to be cloned. Results are written to
`nn-lite-results/` in the folder you run NN-Lite from; see
[Contributing results to the dataset](#contributing-results-to-the-dataset) to add them to LEMUR.

Or install it from source:
```bash
git clone https://github.com/ABrain-One/nn-lite.git
cd nn-lite
pip install -e .
```

If `nn-lite-bench` stops with `ModuleNotFoundError: No module named 'ab.lite'`, your `PYTHONPATH`
includes a checkout of the NN Dataset, whose `ab` folder then hides the installed one
(`python -c "import ab; print(ab.__path__)"` shows which is used). Remove that folder from
`PYTHONPATH`, or run `unset PYTHONPATH`, and try again.

### Working with an nn-dataset checkout

If you commit results to the NN Dataset, or benchmark models that are newer than the installed
package, NN-Lite can read the models from a git checkout of the dataset and write the results
straight into it. When NN-Lite is installed from source, a checkout next to it is found
automatically:
```bash
cd ..
git clone https://github.com/ABrain-One/nn-dataset.git
```
```
your-folder/
├── nn-lite/
└── nn-dataset/
```
An `nn-dataset` checkout in the folder NN-Lite is run from is found automatically as well. A
checkout elsewhere is chosen with `--dataset-root /path/to/nn-dataset` or the `NN_DATASET_ROOT`
environment variable. NN-Lite prints at start-up where it reads the models from and where it
writes the results.

## Connect a phone

1. On the phone, open **Settings → About phone** and tap **Build number** seven times to enable developer options.
2. Open **Settings → Developer options** (on some phones under **System**) and turn on **USB debugging**.
3. Connect the phone to the computer with a USB cable and accept the **Allow USB debugging?** prompt on the phone.
4. Check that the phone is visible:
   ```bash
   adb devices
   ```
   It should be listed with the state `device` (not `unauthorized`). With several phones
   connected, see [Several phones](#several-phones).

Keep the phone charging during long runs. NN-Lite keeps the screen awake and copies the
`benchmark_model` binary to `/data/local/tmp` on the phone automatically.

## Quick start

Benchmark a single model (a few minutes):
```bash
nn-lite-bench --models AirNet
```
From a source checkout, the same command is `python -m ab.lite.torch2tflite --models AirNet`.

NN-Lite converts `AirNet` to FP32 and INT8, times each file on the CPU, GPU and NPU with
20 runs per backend, and writes (into the nn-dataset checkout instead, if one is used):
```
nn-lite-results/ab/nn/stat/run/tflite/fp32/img-classification_cifar-10_acc_AirNet/android_<device>.json
nn-lite-results/ab/nn/stat/run/tflite/int8/img-classification_cifar-10_acc_AirNet/android_<device>.json
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
appended to `_work/benchmark_errors_<device>.log` in the results folder.

Benchmark every model (runs for hours; safe to stop and restart at any time):
```bash
nn-lite-bench
```

| Option | Meaning |
|---|---|
| `--models NAME [NAME ...]` | Only process these models |
| `--android-runs N` | Timed runs per backend (default 20) |
| `--dataset-root PATH` | Read models from this `nn-dataset` checkout and write results into it |
| `--out PATH` | Write results to this folder instead (default: the checkout, or `./nn-lite-results`) |
| `--model-path PATH [PATH ...]` | Benchmark your own models instead (see [Your own models](#your-own-models)) |
| `--serial SERIAL` | Benchmark this phone, as listed by `adb devices` (needed when several are connected) |
| `--restart-every N` | Restart the process after every N models to free memory (default 50; 0 never restarts) |
| `--force` | Forget the connected phone's progress and start from the beginning |
| `--reinstall-bench` | Copy `benchmark_model` to the phone again |

Progress is stored per phone model in `_work/processing_state_<model>.json` of the results folder; models
listed there as processed or failed are skipped when that phone model is benchmarked again,
while a phone of another model starts from the beginning. `--force` resets the progress of the
connected phone model only. Like the result files, progress is identified by the phone model, so
a second phone of the same model continues where the first one left off.

### Several phones

Several phones can be benchmarked at the same time from one computer, each by its own run
of NN-Lite. Choose the phone of each run with `--serial` and the serial number that
`adb devices` lists for it:
```bash
adb devices
# List of devices attached
# R58M12ABCDE    device
# 2A281FDH300    device
nn-lite-bench --serial R58M12ABCDE     # in one terminal
nn-lite-bench --serial 2A281FDH300     # in another terminal
```
Without `--serial` (or the `ANDROID_SERIAL` environment variable), NN-Lite uses the only
phone connected, and stops with the list of phones if there are several. A run only ever
talks to its own phone, even if other phones are connected or reconnected during the run.

The runs can write into the same results folder: the downloads they share are made once,
and each phone has its own temporary folder. Phones of the same model share one progress
file and one result file per model, so they are benchmarked one after the other; a second
run for a phone model that is already being benchmarked stops with an error.

## Your own models

NN-Lite also benchmarks models that are not part of the NN Dataset, without using the dataset at
all. Give the model files, or folders containing them, with `--model-path`:
```bash
nn-lite-bench --model-path my_models/ --calib-dir sample_images/
```
Each model is either

- a **`.pt2` file** saved with [`torch.export`](https://docs.pytorch.org/docs/stable/export.html):
  it needs no Python code, and its input shape is stored in the file. Export the model in
  evaluation mode (`model.eval()`), as the exported graph keeps the mode it was exported in:
  ```python
  torch.export.save(torch.export.export(model.eval(), (torch.randn(1, 3, 224, 224),)), "mymodel.pt2")
  ```
- or a **`.py` file with a `.pt` or `.pth` file of the same name** next to it (`mymodel.py` and
  `mymodel.pth`), holding either the weights (`torch.save(model.state_dict(), ...)`) or the whole
  model (`torch.save(model, ...)`). NN-Lite builds the network with `create_model()` if the `.py`
  file defines one, otherwise with `Net()` or the file's only model class. The input is
  `--input-size` pixels square (default 224).

A `.pt` file alone is not enough: it holds the weights but not the code that defines the network,
so PyTorch cannot rebuild the model from it. Load whole-model files only from sources you trust,
as loading them can run code stored in the file.

FP32 is always benchmarked. INT8 needs sample inputs for calibration: pass a folder of images with
`--calib-dir` (up to 50 are used). They are resized to the model's input and normalised with the
ImageNet statistics, unless the `.py` file defines `input_transform`, a function that turns a PIL
image into a tensor. Results are written to `nn-lite-results/custom/{fp32,int8}/<model>/`, in the
same format as the dataset's records.

## Contributing results to the dataset

The results folder has the same layout as the NN Dataset, so adding your measurements to LEMUR
takes three steps:
```bash
git clone https://github.com/ABrain-One/nn-dataset.git
cp -r nn-lite-results/ab nn-dataset/
cd nn-dataset && git add ab/nn/stat/run/tflite && git commit -m "Add LiteRT benchmarks for <device>"
```
Then open a pull request on the [NN Dataset](https://github.com/ABrain-One/nn-dataset) repository.
The `_work` folder (downloads, progress and logs) is not part of the dataset and is not copied.
From then on, NN-Lite run from the same folder finds this checkout and writes new results
straight into it.

## Optional: emulator path (Android Studio)

The earlier version of NN-Lite runs models inside an Android emulator through the Android app in
`App/`. It uses the NN Dataset Python package installed with NN-Lite; to use its latest
development version instead:
```bash
rm -rf db
pip uninstall -y nn-dataset
pip install --no-cache-dir git+https://github.com/ABrain-One/nn-dataset
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

The unit tests cover output parsing, error extraction, the result schema, the choice of the
model source and of the phone, the locks between runs and the options kept across restarts.
They need neither a phone nor PyTorch, TensorFlow or the NN Dataset:
```bash
pip install pytest filelock
python -m pytest tests
```

## Contributing and support

Bug reports, questions and feature requests are welcome in the
[issue tracker](https://github.com/ABrain-One/nn-lite/issues). See [CONTRIBUTING.md](https://github.com/ABrain-One/nn-lite/blob/main/CONTRIBUTING.md)
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

NN-Lite is released under the [MIT License](https://github.com/ABrain-One/nn-lite/blob/main/LICENSE).

#### The idea and leadership of Dr. Ignatov
