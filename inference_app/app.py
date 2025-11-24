import sys
import os

# Add parent directory to path to import existing modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import gradio as gr
from gradio_wrapper import classify_plant, update_model_choices
import random

def predict_and_retrieve(source_type, local_path, hf_id, pth_file, pth_arch, pth_classes, herbarium_path, input_image):
    # 1. Classify
    try:
        predictions = classify_plant(source_type, local_path, hf_id, pth_file, pth_arch, pth_classes, input_image)
    except Exception as e:
        raise gr.Error(f"Prediction failed: {e}")
    
    if not predictions:
        return "No predictions", None

    # Get top prediction
    top_class = max(predictions, key=predictions.get)
    confidence = predictions[top_class]
    
    result_text = f"Predicted Species: {top_class}\nConfidence: {confidence:.2%}"
    
    # 2. Retrieve Herbarium Images
    herbarium_images = []
    if herbarium_path and os.path.isdir(herbarium_path):
        target_dir = None
        
        # Try exact match
        possible_path = os.path.join(herbarium_path, top_class)
        if os.path.isdir(possible_path):
            target_dir = possible_path
        else:
            # Try normalized match (snake_case vs spaces etc)
            def normalize(n): return str(n).lower().replace(' ', '_').replace('-', '_')
            
            norm_top = normalize(top_class)
            # List all directories in herbarium path
            try:
                for d in os.listdir(herbarium_path):
                    full_d = os.path.join(herbarium_path, d)
                    if os.path.isdir(full_d):
                        if normalize(d) == norm_top:
                            target_dir = full_d
                            break
            except OSError:
                pass
        
        if target_dir:
            # Fetch images
            try:
                files = [os.path.join(target_dir, f) for f in os.listdir(target_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'))]
                # Shuffle and pick a few
                random.shuffle(files)
                herbarium_images = files[:6] # Show top 6 matches
                
                if not herbarium_images:
                    result_text += "\n\n(Class folder found in herbarium dataset, but no images inside)"
            except OSError:
                result_text += "\n\n(Error reading class folder)"
        else:
             result_text += f"\n\n(No matching folder found in herbarium dataset for '{top_class}')"
    elif herbarium_path:
        result_text += "\n\n(Herbarium path provided is invalid)"

    return result_text, herbarium_images

# UI Construction
with gr.Blocks(theme=gr.themes.Soft(), title="Plant Species Identification") as demo:
    gr.Markdown("# Plant Species Identification System")
    gr.Markdown("Upload a field image to identify the species and view matching herbarium specimens.")
    
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### 1. Model Settings")
                inf_source = gr.Radio(
                    choices=["Local", "Hugging Face Hub", "Local .pth"],
                    value="Local",
                    label="Model source"
                )
                
                inf_model_path = gr.Dropdown(
                    label="Select local model", 
                    choices=[], 
                    value=None, 
                    filterable=True,
                    visible=True,
                    allow_custom_value=True
                )
                
                inf_hf_id = gr.Textbox(
                    label="Hugging Face model ID", 
                    visible=False
                )
                
                with gr.Column(visible=False) as inf_pth_group:
                    inf_pth_file = gr.Textbox(label="Path to .pth file")
                    inf_pth_classes = gr.Textbox(label="Path to class list (txt/json)")
                    inf_pth_arch = gr.Textbox(label="Architecture name (timm)")

            with gr.Group():
                gr.Markdown("### 2. Reference Data")
                herbarium_dir = gr.Textbox(
                    label="Herbarium Dataset Directory",
                    placeholder="Path to folder containing class subfolders of herbarium images"
                )

            with gr.Group():
                gr.Markdown("### 3. Input")
                inf_input_image = gr.Image(type="pil", label="Upload Field Image")
                inf_button = gr.Button("Identify Species", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### Results")
            res_text = gr.Textbox(label="Prediction", interactive=False)
            res_gallery = gr.Gallery(label="Matching Herbarium Specimens", columns=3, height="auto")

    # Event Handlers
    def update_inf_inputs(source):
        return (
            gr.update(visible=(source == "Local")),
            gr.update(visible=(source == "Hugging Face Hub")),
            gr.update(visible=(source == "Local .pth"))
        )

    inf_source.change(
        fn=update_inf_inputs,
        inputs=[inf_source],
        outputs=[inf_model_path, inf_hf_id, inf_pth_group]
    )

    inf_button.click(
        fn=predict_and_retrieve,
        inputs=[inf_source, inf_model_path, inf_hf_id, inf_pth_file, inf_pth_arch, inf_pth_classes, herbarium_dir, inf_input_image],
        outputs=[res_text, res_gallery]
    )

    def refresh_models():
        # update_model_choices returns 3 updates (for 3 different dropdowns in the main app).
        # We only have 1 dropdown here, so we take the first one.
        updates = update_model_choices()
        return updates[0]

    demo.load(fn=refresh_models, outputs=inf_model_path)

if __name__ == "__main__":
    demo.launch()
