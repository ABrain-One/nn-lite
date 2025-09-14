# Mobile-Ready Neural Network Verifier


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

## Installing required packages
```bash
pip install -r requirements.txt
```


## Installing/Updating NN Dataset from GitHub:
```bash
rm -rf db
pip uninstall -y nn-dataset
pip install git+https://github.com/ABrain-One/nn-dataset --upgrade --force --extra-index-url https://download.pytorch.org/whl/cu126
```

## How to run the torch2tflite file:
```bash
python create_tflite.py <model1,model2,model3>
Example: python create_tflite.py resnet50,mobilenet,efficientnet
Or single model: python create_tflite.py AirNet
Or all models: python create_tflite.py all
```
