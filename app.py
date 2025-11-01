import gradio as gr
import numpy as np
from PIL import Image
import joblib
import torch
import torch.nn as nn
from timm.models import create_model
from Src.feature_extractor import DinoV2FeatureExtractor
import os

# --- Configuration ---
DEFAULT_FEATURE_DIR = os.path.join("Dataset", "features", "train")

# --- Model Loading ---
def load_joblib_model(model_path):
    """
    Loads a scikit-learn model from a .joblib file.
    """
    print(f"Loading joblib model from {model_path}...")
    ml_model = joblib.load(model_path)
    feature_extractor = DinoV2FeatureExtractor()
    print("Model loaded.")
    return ml_model, feature_extractor

def load_pytorch_model(model_path, num_classes):
    """
    Loads a PyTorch model (linear probe) and the DINOv2 backbone.
    """
    print(f"Loading PyTorch model from {model_path}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load DINOv2 backbone
    dino_model = create_model('dinov2_vitb14_reg', pretrained=True, num_classes=0).to(device)
    dino_model.eval()

    # Load the trained linear classifier
    classifier = nn.Linear(dino_model.embed_dim, num_classes)
    classifier.load_state_dict(torch.load(model_path, map_location=device))
    classifier.to(device)
    classifier.eval()

    print("PyTorch model loaded.")
    return dino_model, classifier

# --- Prediction Functions ---
def predict_joblib(image, ml_model, feature_extractor, class_names):
    """
    Predicts using the joblib-loaded model.
    """
    image_path = "temp_image.jpg"
    image.save(image_path)
    features = feature_extractor.extract_features(image_path)
    os.remove(image_path)

    probabilities = ml_model.predict_proba(features.reshape(1, -1))[0]
    top5_indices = np.argsort(probabilities)[-5:][::-1]
    top5_predictions = {class_names[i]: float(probabilities[i]) for i in top5_indices}
    return top5_predictions

def predict_pytorch(image, dino_model, classifier, class_names):
    """
    Predicts using the PyTorch model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = create_transform(**resolve_data_config(dino_model.default_cfg))
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = dino_model(image_tensor)
        outputs = classifier(features)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]

    top5_indices = torch.argsort(probabilities, descending=True)[:5]
    top5_predictions = {class_names[i]: float(probabilities[i]) for i in top5_indices}
    return top5_predictions

# --- Gradio Interface ---
def create_gradio_app(model_paths):
    """
    Creates the Gradio web application.
    """
    try:
        class_names = sorted(os.listdir(DEFAULT_FEATURE_DIR))
        num_classes = len(class_names)
    except FileNotFoundError:
        print(f"Error: Feature directory not found at {DEFAULT_FEATURE_DIR}")
        class_names = [f"class_{i}" for i in range(100)] # Placeholder
        num_classes = len(class_names)

    def gradio_predict_wrapper(model_choice, image):
        model_info = model_paths[model_choice]
        model_path = model_info["path"]
        model_type = model_info["type"]

        if model_type == "joblib":
            ml_model, feature_extractor = load_joblib_model(model_path)
            return predict_joblib(image, ml_model, feature_extractor, class_names)
        elif model_type == "pytorch":
            dino_model, classifier = load_pytorch_model(model_path, num_classes)
            return predict_pytorch(image, dino_model, classifier, class_names)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    model_choices = list(model_paths.keys())

    iface = gr.Interface(
        fn=gradio_predict_wrapper,
        inputs=[
            gr.Dropdown(choices=model_choices, label="Choose Model"),
            gr.Image(type="pil", label="Upload a plant image")
        ],
        outputs=gr.Label(num_top_classes=5, label="Top 5 Predictions"),
        title="Cross-Domain Plant Identification",
        description="Upload an image of a plant to classify it.",
        examples=[
            [model_choices[0], "./Dataset/validation/105951/0ce1c2b5630236a87218a399a0336842.jpg"],
        ]
    )
    return iface

if __name__ == "__main__":
    model_paths = {
        "DINOv2 Linear Probe": {
            "path": os.path.join("results", "DINOv2_Linear_Probe", "dinov2_linear_probe.pth"),
            "type": "pytorch"
        },
        "DINOv2 + SVM (GridSearch)": {
            "path": os.path.join("results", "DINOv2_FeatureExtractor_SVM_GridSearch", "svm_gridsearch_model.joblib"),
            "type": "joblib"
        },
        "DINOv2 + Random Forest": {
            "path": os.path.join("results", "DINOv2_FeatureExtractor_RF", "random_forest_model.joblib"),
            "type": "joblib"
        },
    }

    app = create_gradio_app(model_paths)
    app.launch(share=True)