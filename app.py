
import gradio as gr
import numpy as np
from PIL import Image
import joblib
from Src.feature_extractor import DinoV2FeatureExtractor
import os

# --- Configuration ---
DEFAULT_MODEL_PATH = os.path.join("results", "DINOv2_FeatureExtractor_SVM_GridSearch", "svm_gridsearch_model.joblib")
DEFAULT_FEATURE_DIR = os.path.join("Dataset", "features", "train")

# --- Load Models ---
def load_models(model_path):
    """
    Loads the ML model and the feature extractor.
    """
    print("Loading models...")
    ml_model = joblib.load(model_path)
    feature_extractor = DinoV2FeatureExtractor()
    print("Models loaded.")
    return ml_model, feature_extractor

# --- Prediction Function ---
def predict(image, ml_model, feature_extractor, class_names):
    """
    Predicts the class of an image.
    """
    # Save the uploaded image temporarily
    image_path = "temp_image.jpg"
    image.save(image_path)

    # Extract features
    features = feature_extractor.extract_features(image_path)

    # Predict probabilities
    probabilities = ml_model.predict_proba(features.reshape(1, -1))[0]

    # Get top 5 predictions
    top5_indices = np.argsort(probabilities)[-5:][::-1]
    top5_predictions = {class_names[i]: float(probabilities[i]) for i in top5_indices}

    # Clean up the temporary image file
    os.remove(image_path)

    return top5_predictions

# --- Gradio Interface ---
def create_gradio_app(model_paths):
    """
    Creates the Gradio web application.
    """
    # Load class names from the feature directory structure
    try:
        class_names = sorted(os.listdir(DEFAULT_FEATURE_DIR))
    except FileNotFoundError:
        print(f"Error: Feature directory not found at {DEFAULT_FEATURE_DIR}")
        print("Please run the feature extraction script first.")
        class_names = ["class_1", "class_2", "class_3", "class_4", "class_5"] # Placeholder

    def gradio_predict_wrapper(model_choice, image):
        model_path = model_paths[model_choice]
        ml_model, feature_extractor = load_models(model_path)
        return predict(image, ml_model, feature_extractor, class_names)

    # Define the model choices for the dropdown
    model_choices = list(model_paths.keys())

    # Create the Gradio interface
    iface = gr.Interface(
        fn=gradio_predict_wrapper,
        inputs=[
            gr.Dropdown(choices=model_choices, label="Choose Model"),
            gr.Image(type="pil", label="Upload a plant image")
        ],
        outputs=gr.Label(num_top_classes=5, label="Top 5 Predictions"),
        title="Cross-Domain Plant Identification",
        description="Upload an image of a plant to classify it using a DINOv2 feature extractor and a machine learning model.",
        examples=[
            [model_choices[0], "./Dataset/validation/105951/0ce1c2b5630236a87218a399a0336842.jpg"],
            [model_choices[0], "./Dataset/validation/12254/0a6548a656b34a659a0963b3693d4836.jpg"],
        ]
    )
    return iface

if __name__ == "__main__":
    # --- Define Model Paths ---
    # This dictionary allows you to add more models to the dropdown easily
    model_paths = {
        "DINOv2 + SVM (GridSearch)": os.path.join("results", "DINOv2_FeatureExtractor_SVM_GridSearch", "svm_gridsearch_model.joblib"),
        "DINOv2 + Random Forest": os.path.join("results", "DINOv2_FeatureExtractor_RF", "random_forest_model.joblib"),
    }

    # Create and launch the Gradio app
    app = create_gradio_app(model_paths)
    app.launch(share=True)
