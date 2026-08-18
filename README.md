# Mobile-Ready AI: Verification and Deployment on Edge Devices

<img src='https://abrain.one/img/nnlite-logo.png' width='25%'/>

The original open-source version of the <a href='https://github.com/ABrain-One/NN-Lite/'>NN Lite</a> was developed by <strong>Faraz Kayani</strong>, <strong>Saif U Din</strong> and <strong>Muhammad Ahsan Hussain</strong> at the Computer Vision Laboratory, University of Würzburg, Germany, under the supervision and technical guidance of <strong>Dr. Dmitry Ignatov</strong>, whose foundational work established the basis for the project.

## Create and Activate a Virtual Environment (recommended)
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

## Installing requirements 
```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
```

## Install/Update NN Dataset from GitHub:
```bash
rm -rf db
pip install --no-cache-dir git+https://github.com/ABrain-One/nn-dataset --upgrade --force --extra-index-url https://download.pytorch.org/whl/cu126
```

## Install Android Studio (Outside of Virtual Environment) 'Android Studio Narwhal 3 Feature Drop | 2025.1.3' through a ready made script: 
For Linux:
```bash
chmod +x install-android-studio.sh
./install-android-studio.sh
```

# Run all models (original behavior)
```bash
python -m ab.lite.torch2tflite-all
```

# Run single model
```bash
python -m ab.lite.torch2tflite-all AirNet
```
	
# Run multiple models as separate arguments
```bash
python -m ab.lite.torch2tflite-all AirNet ga-196 ga-197 ga-198	
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
	
# Open the App in Android Studio:	
	
Select 'App' and import it as a Project
Go to Tools->Device Manager->Add a new device through '+' symbol. e.g. Pixel 5

	
# Set up Android SDK Environment Variables:

Select 'App' and import it as a Project
Go to Tools->Device Manager->Add a new device through '+' symbol. e.g. Pixel 5
	
# Set up Android SDK Environment Variables:
```bash
nano ~/.bashrc
```
Go to the end of the file and add these 3 lines of code according to your available paths (below is the example path): 
		export ANDROID_SDK_ROOT="/home/ahsan/Android/Sdk"
		export ANDROID_HOME="/home/ahsan/Android/Sdk"
		export PATH="$PATH:/home/ahsan/Android/Sdk/cmdline-tools/latest/bin:/home/ahsan/.local/bin"
	You can find your paths through: Tools->Device Manager->Android SDK Location

## Citation

If you find this project to be useful for your research, please consider citing our articles:
```bibtex
@article{ABrain.NN-Lite,
    title = {AI on the Edge: An Automated Pipeline for PyTorch-to-Android Deployment and Benchmarking},
	author = {Saif U Din and Muhammad Ahsan Hussain and Mohsin Ikram and Faraz Kiyani and Dmitry Ignatov and Radu Timofte},
	doi = {10.20944/preprints202511.1831.v1},
	url = {https://doi.org/10.20944/preprints202511.1831.v1},
	year = 2025,
	month = {November},
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

#### The idea and leadership of Dr. Ignatov
