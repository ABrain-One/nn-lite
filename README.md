
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

# Simply install the latest version of Android Studio.

Android Studio: Required to install the Android SDK and create an emulator. Emolator can be installed by the script, but it takes time better to install it yourself so that you can have a UI status of the remaining installation. 
Go to Device Manager, click on the plus icon and select a device, and simply install it. 

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
