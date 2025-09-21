Automated TFLite Model Benchmarking Suite
This project provides a fully automated, one-command solution to benchmark PyTorch models on an Android emulator. The system generates a .tflite file from a model in the database, automatically launches an emulator, runs the model inference, and generates a clean JSON report with performance metrics.

Prerequisites
Before you begin, ensure you have the following installed and configured on your Linux system:

Python 3.10+

Android Studio: Required to install the Android SDK and create an emulator.

An Android Virtual Device (AVD): You must create at least one emulator using the AVD Manager in Android Studio.

Configured System PATH: The Android SDK tools must be accessible from your terminal.

To do this permanently, add the following lines to the end of your ~/.bashrc or ~/.zshrc file (adjust the path if your SDK is located elsewhere):

# Android SDK Environment Variables
export ANDROID_HOME=$HOME/Android/Sdk
export PATH=$PATH:$ANDROID_HOME/platform-tools
export PATH=$PATH:$ANDROID_HOME/emulator

After saving the file, apply the changes by running source ~/.bashrc or by opening a new terminal.

Setup Instructions
Follow these simple, one-time steps to prepare the project.

1. Clone the Repository

git clone <your_repository_url>
cd nn-lite

2. Set Up the Python Environment
This command creates an isolated Python environment and installs all the necessary libraries.

# Create the virtual environment folder
python3 -m venv .venv

# Install the required packages from requirements.txt
./.venv/bin/pip install -r requirements.txt

3. Make the Main Script Executable
This command gives your system permission to run the automation script.

chmod +x run_everything.sh

How to Run the Benchmark
This is the single command to run the entire automated process. You must activate the virtual environment first.

1. Activate the Virtual Environment

source .venv/bin/activate

You will see (.venv) appear at the start of your command prompt.

2. Run the Script
Provide the names of the models you wish to test as arguments.

To run a single model:

./run_everything.sh AirNext

To run multiple models in sequence:

./run_everything.sh AirNext resnet50 Bagnet

