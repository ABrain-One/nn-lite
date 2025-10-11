
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
## Install/Update NN Dataset from GitHub:
	```bash
	rm -rf db
	pip uninstall -y nn-dataset
	pip install git+https://github.com/ABrain-One/nn-dataset --upgrade --force --extra-index-url https://download.pytorch.org/whl/cu126
	```

## Installing/Updating requirments 
	```bash
	pip install -r requirements.txt
	```

## Install Android Studio (Outside of Virtual Enviroment) 'Android Studio Narwhal 3 Feature Drop | 2025.1.3' through a ready made script: 
For Linux:
	```bash
	chmod +x install-android-studio-local.sh
	./install-android-studio-local.sh
	```

# Run all models (original behavior)
	```bash
	python torch2tflite-all.py
	```

# Run single model
	```bash
	python torch2tflite-all.py AirNet
	```
	
# Run multiple models as separate arguments
	```bash
	python torch2tflite-all.py AirNet ga-196 ga-197 ga-198	
	```
		

## OR Install Android Studio 'Android Studio Narwhal 3 Feature Drop | 2025.1.3' manually:

# Download Link for 'Android Studio Narwhal 3 Feature Drop | 2025.1.3':
	https://developer.android.com/studio/archive
	
# Install the Android Studio:
	```bash
	sudo apt update
	sudo apt install openjdk-17-jdk
	cd ~/Downloads
	unzip android-studio-*.zip
	sudo mv android-studio /opt/
	```
	
# Launch the Android Studio:
	```bash
	/opt/android-studio/bin/studio.sh
	```
	
# Open the Kotlinapplication in Android Studio:	
	```bash
	Select 'Kotlinapplication' and import it as a Project
	Go to Tools->Device Manager->Add a new device through '+' symbol. e.g. Pixel 5
	```
	
# Set up Android SDK Environment Variables:
	```bash
	nano ~/.bashrc
	Go to the end of the file and add these 3 lines of code according to your available paths (below is the example path): 
		export ANDROID_SDK_ROOT="/home/ahsan/Android/Sdk"
		export ANDROID_HOME="/home/ahsan/Android/Sdk"
		export PATH="$PATH:/home/ahsan/Android/Sdk/cmdline-tools/latest/bin:/home/ahsan/.local/bin"
	You can find your paths through: Tools->Device Manager->Android SDK Location
	```

# Run all models (original behavior)
	```bash
	python torch2tflite-all.py
	```

# Run single model
	```bash
	python torch2tflite-all.py AirNet
	```
	
# Run multiple models as separate arguments
	```bash
	python torch2tflite-all.py AirNet ga-196 ga-197 ga-198	
	```

