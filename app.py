import streamlit as st
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

st.set_page_config(
    page_title="PneumaScan",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="auto"
)

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the paths to your pre-trained models
model_paths = {
    "densenet121": "models/densenet121_best.pth",
    "resnet50": "models/resnet50_best.pth",
    "inception_v3": "models/inception_v3_best.pth",
}

# Define the class labels
class_labels = ["Normal", "Pneumonia"]

# Define transformations
transform_224 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_299 = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to load a model
@st.cache_resource
def load_model(model_name, model_path, num_classes=2):
    if model_name == "densenet121":
        model = models.densenet121(pretrained=False)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "inception_v3":
        model = models.inception_v3(pretrained=False, aux_logits=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Load models
models_dict = {name: load_model(name, path) for name, path in model_paths.items()}

# Function to predict using a single model
def predict_single_model(model, image_tensor):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
    return predicted.item()

# Load image
def load_image(image, model_name):
    transform = transform_299 if model_name == "inception_v3" else transform_224
    image = Image.open(image).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor

# Ensemble prediction
def predict_ensemble(image):
    predictions = []
    for model_name, model in models_dict.items():
        image_tensor = load_image(image, model_name)
        prediction = predict_single_model(model, image_tensor)
        predictions.append(prediction)

    ensemble_prediction = np.bincount(predictions).argmax()
    return class_labels[ensemble_prediction], predictions

# Streamlit UI
st.title("🩺PneumaScan🩻 - Pneumonia Detection App🫁")
st.write("This app uses the x-ray image of the patient to detect whether the patient has Pneumonia or not. Upload an image to get the prediction!")

# Function to validate whether the image is a chest X-ray
def is_xray_image(image):
    image = Image.open(image).convert("RGB")  # Ensure the image is RGB
    grayscale_image = image.convert("L")  # Convert to grayscale

    # Calculate average brightness (mean pixel value)
    brightness = np.array(grayscale_image).mean()

    # Check if the image is predominantly grayscale
    rgb_diff = np.array(image) - np.array(grayscale_image)[:, :, None]
    max_diff = np.max(np.abs(rgb_diff))

    # Conditions for determining if it's likely an X-ray
    is_grayscale = max_diff < 30  # Allow small differences

    return is_grayscale

# Update the Streamlit UI with validation
uploaded_file = st.file_uploader("Upload an image", type=["jpeg", "jpg", "png"])
if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    if st.button("Classify Image"):
        with st.spinner("Validating..."):
            if is_xray_image(uploaded_file):
                ensemble_class, individual_predictions = predict_ensemble(uploaded_file)
                st.success(f"Prediction: {ensemble_class}")
            else:
                st.error("Error: The uploaded image does not appear to be a valid chest X-ray.")
