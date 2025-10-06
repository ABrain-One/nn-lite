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

## Installing/Updating requirments 

```
pip install -r requirements.txt

```



## Installing/Updating Android Studio

Simply Install the latest version of android studio.

Android Studio: Required to install the Android SDK and create an emulator. emolator can be install by the script but it take time better to install it yourself so that you can have UI status of remaining installation. 
Go to device manager and click on plus icon and select a device and simply install it. 

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


