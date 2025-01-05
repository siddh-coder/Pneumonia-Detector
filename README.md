# Pneumonia Detection Web Application

## Overview
This project is a web-based application designed to detect pneumonia from chest X-ray images using pre-trained deep learning models. The application supports uploading images via a user-friendly interface and provides predictions from three models (`DenseNet121`, `ResNet50`, and `Inception_v3`) along with an ensemble prediction based on majority voting.

The application is built using:
- **Flask** for the backend
- **Streamlit** for the frontend
- **PyTorch** for deep learning model inference

## Features
- Upload chest X-ray images directly through the web interface.
- Perform inference using three pre-trained models.
- Ensemble prediction based on majority voting from individual model outputs.
- Works entirely on CPU, ensuring compatibility with systems without GPU.

## Requirements

### Python Version
- Python 3.8 or higher

### Dependencies
The dependencies are listed in `requirements.txt`. Install them using:
```bash
pip install -r requirements.txt
```

### Key Packages
- Flask
- torch
- torchvision
- Pillow
- numpy

## Installation and Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/pneumonia-detection-web-app.git
   cd pneumonia-detection-web-app
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # For Windows: env\Scripts\activate
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Make sure the pre-trained models are in the project models directory. Ensure the filenames match:
   - `models/densenet121_best.pth`
   - `models/resnet50_best.pth`
   - `models/inception_v3_best.pth`

5. Run the application:
   ```bash
   streamlit run app.py
   ```

6. Open the application in your web browser at `http://127.0.0.1:5000`.


## Usage
1. Open the web application in your browser.
2. Upload a chest X-ray image (JPEG or PNG format).
3. View the ensemble result.


## Contributing
Contributions are welcome! Please open an issue or submit a pull request for improvements or new features.

## Acknowledgments
- Pre-trained models were sourced from the PyTorch library.
- Inspired by medical image analysis research and applications.
- The dataset used for training and fine-tuning the models is of kaggle of whose link is provded below:
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
---


