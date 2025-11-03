"""
Gradio Web Application for Cross-Domain Plant Species Classification

This application provides a user-friendly web interface for classifying plant species
using trained DINOv2 models from both Approach A (Feature Extraction) and
Approach B (Fine-Tuning).

Features:
- Model selector dropdown with all trained models
- Image upload for plant identification
- Top-5 predictions with species names and confidence scores
- Model performance metrics display

Usage:
    python app.py
"""

import gradio as gr
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import json
import joblib
import timm
import numpy as np
from functools import lru_cache


# Configuration
CLASSES_FILE = 'classes.txt'
APPROACH_A_DIR = 'Approach_A_Feature_Extraction'
APPROACH_B_DIR = 'Approach_B_Fine_Tuning/Models'


def load_class_names():
    """Load plant species names"""
    if os.path.exists(CLASSES_FILE):
        with open(CLASSES_FILE, 'r') as f:
            return [line.strip() for line in f.readlines()]
    return [f"Class {i}" for i in range(100)]


CLASS_NAMES = load_class_names()


class LinearProbe(nn.Module):
    """Linear probe model for Approach A"""
    def __init__(self, input_dim, num_classes):
        super(LinearProbe, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def find_all_models():
    """Find all trained models from both approaches"""
    models = {}

    # Approach A: Feature extraction + traditional ML
    if os.path.exists(APPROACH_A_DIR):
        results_dir = os.path.join(APPROACH_A_DIR, 'results')
        if os.path.exists(results_dir):
            for model_name in os.listdir(results_dir):
                model_path = os.path.join(results_dir, model_name)
                if os.path.isdir(model_path):
                    sklearn_model = os.path.join(model_path, 'best_model.joblib')
                    pytorch_model = os.path.join(model_path, 'best_model.pth')
                    config_file = os.path.join(model_path, 'training_config.json')

                    if os.path.exists(sklearn_model):
                        models[f"Approach_A | {model_name}"] = {
                            'approach': 'A',
                            'type': 'sklearn',
                            'model_path': sklearn_model,
                            'config_path': config_file,
                            'name': model_name
                        }
                    elif os.path.exists(pytorch_model) and os.path.exists(config_file):
                        models[f"Approach_A | {model_name}"] = {
                            'approach': 'A',
                            'type': 'pytorch_linear',
                            'model_path': pytorch_model,
                            'config_path': config_file,
                            'name': model_name
                        }

    # Approach B: Fine-tuned models
    if os.path.exists(APPROACH_B_DIR):
        for model_name in os.listdir(APPROACH_B_DIR):
            model_path = os.path.join(APPROACH_B_DIR, model_name)
            if os.path.isdir(model_path):
                pytorch_model = os.path.join(model_path, 'best_model.pth')
                config_file = os.path.join(model_path, 'training_config.json')

                if os.path.exists(pytorch_model) and os.path.exists(config_file):
                    models[f"Approach_B | {model_name}"] = {
                        'approach': 'B',
                        'type': 'pytorch_full',
                        'model_path': pytorch_model,
                        'config_path': config_file,
                        'name': model_name
                    }

    return models


ALL_MODELS = find_all_models()


def get_image_transform():
    """Get image preprocessing transform"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


@lru_cache(maxsize=5)
def load_feature_extractor(model_type):
    """Load DINOv2 feature extractor for Approach A"""
    model_configs = {
        'plant_pretrained_base': 'vit_base_patch14_reg4_dinov2.lvd142m',
        'imagenet_small': 'vit_small_patch14_reg4_dinov2.lvd142m',
        'imagenet_base': 'vit_base_patch14_reg4_dinov2.lvd142m',
        'imagenet_large': 'vit_large_patch14_reg4_dinov2.lvd142m'
    }

    model_name = model_configs.get(model_type, 'vit_base_patch14_reg4_dinov2.lvd142m')
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model.eval()

    return model


@lru_cache(maxsize=5)
def load_model(model_key):
    """Load a model with caching"""
    if model_key not in ALL_MODELS:
        return None, None

    model_info = ALL_MODELS[model_key]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        if model_info['type'] == 'sklearn':
            # Load sklearn model (SVM, RF)
            model = joblib.load(model_info['model_path'])
            config = None
            if os.path.exists(model_info['config_path']):
                with open(model_info['config_path'], 'r') as f:
                    config = json.load(f)
            return model, config

        elif model_info['type'] == 'pytorch_linear':
            # Load linear probe
            with open(model_info['config_path'], 'r') as f:
                config = json.load(f)

            model = LinearProbe(config['input_dim'], config['num_classes'])
            model.load_state_dict(torch.load(model_info['model_path'], map_location=device))
            model = model.to(device)
            model.eval()

            return model, config

        elif model_info['type'] == 'pytorch_full':
            # Load fine-tuned model
            with open(model_info['config_path'], 'r') as f:
                config = json.load(f)

            model = timm.create_model(
                config['model_name'],
                pretrained=False,
                num_classes=config['num_classes']
            )
            model.load_state_dict(torch.load(model_info['model_path'], map_location=device))
            model = model.to(device)
            model.eval()

            return model, config

    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

    return None, None


def predict(image, model_key):
    """
    Predict plant species from an uploaded image.

    Args:
        image: PIL Image uploaded by user
        model_key: Selected model key

    Returns:
        dict: Top-5 predictions with confidence scores
        str: Model information HTML
    """
    if model_key not in ALL_MODELS:
        return {"Error": "Please select a model"}, "No model selected"

    model_info = ALL_MODELS[model_key]

    try:
        # Load model
        model, config = load_model(model_key)

        if model is None:
            return {"Error": "Failed to load model"}, "Model loading failed"

        # Preprocess image
        transform = get_image_transform()
        image_rgb = image.convert("RGB")
        image_tensor = transform(image_rgb).unsqueeze(0)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Predict based on approach
        if model_info['approach'] == 'A':
            # Approach A: Extract features then classify
            # Extract features
            feature_type = model_info['name'].split('_', 1)[1]  # e.g., "imagenet_base"
            feature_extractor = load_feature_extractor(feature_type)
            feature_extractor = feature_extractor.to(device)

            with torch.no_grad():
                features = feature_extractor(image_tensor.to(device))
                features = features.cpu().numpy()

            if model_info['type'] == 'sklearn':
                # SVM or Random Forest
                probs = model.predict_proba(features)[0]
            else:
                # Linear probe
                features_tensor = torch.from_numpy(features).float().to(device)
                with torch.no_grad():
                    outputs = model(features_tensor)
                    probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

        else:
            # Approach B: Direct prediction
            with torch.no_grad():
                outputs = model(image_tensor.to(device))
                probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

        # Get top 5 predictions
        top5_indices = np.argsort(probs)[-5:][::-1]
        top5_probs = probs[top5_indices]

        # Create results dictionary
        results = {}
        for idx, prob in zip(top5_indices, top5_probs):
            species_name = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"Class {idx}"
            results[f"#{idx}: {species_name}"] = float(prob)

        # Model info
        info_html = f"""
        <div style='padding: 15px; background: #f0f0f0; border-radius: 10px;'>
            <h3 style='margin-top: 0;'>Model Information</h3>
            <p><b>Approach:</b> {model_info['approach']}</p>
            <p><b>Model:</b> {model_info['name']}</p>
        """

        if config:
            if 'best_val_accuracy' in config:
                info_html += f"<p><b>Validation Accuracy:</b> {config['best_val_accuracy']:.2f}%</p>"
            if 'best_cv_score' in config:
                info_html += f"<p><b>CV Score:</b> {config['best_cv_score']:.4f}</p>"

        info_html += "</div>"

        return results, info_html

    except Exception as e:
        return {"Error": str(e)}, f"<p style='color: red;'>Error: {str(e)}</p>"


# Custom CSS
css = """
.gradio-container {
    background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%) !important;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
}
.title {
    text-align: center;
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    padding: 30px;
    margin-bottom: 30px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
}
.description {
    text-align: center;
    background: rgba(255, 255, 255, 0.95);
    border-radius: 15px;
    padding: 20px;
    margin: 20px 0;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    color: #2c3e50;
}
"""

# Create Gradio interface
with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
    # Title
    gr.HTML("""
    <div class="title">
        <h1 style="color: #27ae60; font-size: 2.5em; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
            🌱 Cross-Domain Plant Species Classification
        </h1>
        <p style="color: white; font-size: 1.2em; margin: 10px 0 0 0; opacity: 0.9;">
            Powered by DINOv2 Models
        </p>
    </div>
    """)

    # Description
    gr.HTML("""
    <div class="description">
        <p style="margin: 0; font-size: 1.1em; color: black;">
            Upload a plant image to identify the species using state-of-the-art DINOv2 models.
            Choose between feature extraction (Approach A) or fine-tuned models (Approach B).
        </p>
    </div>
    """)

    # Main content
    with gr.Row():
        # Left column
        with gr.Column(scale=1):
            # Model selector
            model_dropdown = gr.Dropdown(
                choices=list(ALL_MODELS.keys()),
                label="🤖 Select Model",
                info="Choose a trained model for prediction",
                value=list(ALL_MODELS.keys())[0] if ALL_MODELS else None
            )

            # Image upload
            image_input = gr.Image(
                type="pil",
                label="📸 Upload Plant Image",
                height=400
            )

            # Buttons
            with gr.Row():
                submit_btn = gr.Button("🔍 Classify", variant="primary", size="lg")
                clear_btn = gr.Button("🗑️ Clear", variant="secondary", size="lg")

        # Right column
        with gr.Column(scale=1):
            # Predictions
            output_label = gr.Label(
                num_top_classes=5,
                label="🏆 Top 5 Predictions",
                show_label=True
            )

            # Model info
            model_info = gr.HTML(label="ℹ️ Model Information")

    # Event handlers
    submit_btn.click(
        fn=predict,
        inputs=[image_input, model_dropdown],
        outputs=[output_label, model_info]
    )

    clear_btn.click(
        fn=lambda: (None, {}, ""),
        inputs=None,
        outputs=[image_input, output_label, model_info]
    )

    # Footer
    gr.HTML("""
    <div style='text-align: center; padding: 20px; color: white; opacity: 0.8;'>
        <p>🌿 Baseline Approach 2: DINOv2 for Cross-Domain Plant Identification</p>
        <p>COS30082 Applied Machine Learning Project</p>
    </div>
    """)


if __name__ == "__main__":
    if not ALL_MODELS:
        print("⚠️  Warning: No trained models found!")
        print("   Train models using Approach A and/or Approach B scripts first.")
        print("\n   Available models will appear in the dropdown once trained.")

    print(f"\n🚀 Starting Gradio app...")
    print(f"   Found {len(ALL_MODELS)} trained models")

    demo.launch(share=True)
