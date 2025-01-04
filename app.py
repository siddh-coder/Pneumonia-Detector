import os
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, flash

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "supersecretkey"

# Define device (CPU-only)
device = torch.device("cpu")

# Define model paths
model_paths = {
    "densenet121": "models/densenet121_best.pth",
    "resnet50": "models/resnet50_best.pth",
    "inception_v3": "models/inception_v3_best.pth",
}

# Define class labels
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

# Load models
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

models_dict = {name: load_model(name, path) for name, path in model_paths.items()}

# Function to load image
def load_image(image, model_name):
    transform = transform_299 if model_name == "inception_v3" else transform_224
    image = image.convert("RGB")
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor

# Function to predict with a single model
def predict_single_model(model, image_tensor):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
    return predicted.item()

# Predict ensemble
def predict_ensemble(image):
    predictions = []
    for model_name, model in models_dict.items():
        image_tensor = load_image(image, model_name)
        prediction = predict_single_model(model, image_tensor)
        predictions.append(prediction)
    ensemble_prediction = np.bincount(predictions).argmax()
    return class_labels[ensemble_prediction], predictions

# Routes
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)
        if file:
            image = Image.open(file.stream)
            ensemble_prediction, individual_predictions = predict_ensemble(image)
            return render_template(
                "index.html",
                prediction=ensemble_prediction,
                individual_predictions=individual_predictions,
            )
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)

