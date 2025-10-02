
## Create and Activate a Virtual Environment (recommended)
For Linux/Mac:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
For Windows:
   ```bash
   python3 -m venv .venv
   .venv\Scripts\activate
   ```
## Installing/Updating NN Dataset from GitHub:
```bash
rm -rf db
pip uninstall -y nn-dataset
pip install git+https://github.com/ABrain-One/nn-dataset --upgrade --force --extra-index-url https://download.pytorch.org/whl/cu126
```

## Installing/Updating Android Studio

Simply Install the latest version of android studio.

Android Studio: Required to install the Android SDK and create an emulator. emolator can be install by the script but it take time better to install it yourself so that you can have UI status of remaining installation. 
Go to device manager and click on plus icon and select a device and simply install it. 
=======
<img src='https://abrain.one/img/nnlite-logo.png' width='50%'/>

Automated TFLite Model Benchmarking Suite
This project provides a fully automated, one-command solution to benchmark PyTorch models on an Android emulator. The system generates a .tflite file from a model in the database, automatically launches an emulator, runs the model inference, and generates a clean JSON report with performance metrics.

Prerequisites
Before you begin, ensure you have the following installed and configured on your Linux system:

Python 3.10+

Android Studio: Required to install the Android SDK and create an emulator.

An Android Virtual Device (AVD): You must create at least one emulator using the AVD Manager in Android Studi

Configured System PATH: The Android SDK tools must be accessible from your terminal.
To do this permanently, add the following lines to the end of your ~/.bashrc or ~/.zshrc file (adjust the path if your SDK is located elsewhere):

# Android SDK Environment Variables
```
export ANDROID_HOME=$HOME/Android/Sdk
export PATH=$PATH:$ANDROID_HOME/platform-tools
export PATH=$PATH:$ANDROID_HOME/emulator
```


# Run all models (original behavior)
python torch2tflite-all.py

# Run single model
python torch2tflite-all.py AirNet

# Run multiple models as separate arguments
python torch2tflite-all.py AirNet ga-196 ga-197 ga-198
