import os
import gradio as gr
import random
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoConfig
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json

def get_model_choices():
    """
    Finds model directories. Prioritises the root directory if it contains model weights.
    Otherwise, falls back to the latest checkpoint.
    """
    choices = []
    try:
        app_dir = os.path.dirname(os.path.abspath(__file__))
        model_parent_dirs = [d for d in os.listdir(app_dir) if os.path.isdir(os.path.join(app_dir, d)) and d.startswith('Model-')]

        for model_dir_name in model_parent_dirs:
            model_dir_path = os.path.join(app_dir, model_dir_name)
            
            # Check for weights in root
            has_weights = os.path.exists(os.path.join(model_dir_path, "model.safetensors")) or \
                          os.path.exists(os.path.join(model_dir_path, "pytorch_model.bin"))
            
            if has_weights:
                choices.append((model_dir_name, model_dir_path))
                continue

            # Fallback to checkpoints
            checkpoints = []
            for item in os.listdir(model_dir_path):
                path = os.path.join(model_dir_path, item)
                if os.path.isdir(path) and item.startswith('checkpoint-'):
                    try:
                        step = int(item.split('-')[-1])
                        checkpoints.append((step, path))
                    except (ValueError, IndexError):
                        continue
            
            if checkpoints:
                latest_checkpoint = sorted(checkpoints, key=lambda x: x[0], reverse=True)[0]
                step, path = latest_checkpoint
                display_name = f"{model_dir_name} (ckpt-{step})"
                choices.append((display_name, path))

    except FileNotFoundError:
        print("Warning: Could not find the app directory to scan for models.")
    
    return sorted(choices, key=lambda x: x[0])

def update_model_choices(current_model=None):
    """Refreshes the list of available models in the dropdowns."""
    choices = get_model_choices()
    choice_values = [c[1] for c in choices]
    
    new_value = None
    if current_model and current_model in choice_values:
        new_value = current_model
    
    return gr.update(choices=choices, value=new_value)

def load_model_generic(source_type, local_path, hf_id, pth_file, pth_arch, pth_classes):
    """Helper to load models from various sources."""
    model = None
    processor = None
    model_type = "hf" # or "timm"
    extra_data = {} # e.g. class_names for timm

    # --- CASE 1: Local or Hugging Face Hub ---
    if source_type in ["Local", "Hugging Face Hub"]:
        model_id = local_path if source_type == "Local" else hf_id
        
        if not model_id:
            raise gr.Error(f"Please specify the {source_type} model.")

        # Handle Local checkpoint logic
        if source_type == "Local":
            if os.path.isfile(model_id):
                model_id = os.path.dirname(model_id)
            if os.path.basename(model_id).startswith('checkpoint-'):
                checkpoint_dir = model_id
                model_id = os.path.dirname(model_id) # Root for processor
            else:
                # Check if root has weights
                has_weights = os.path.exists(os.path.join(model_id, "model.safetensors")) or \
                              os.path.exists(os.path.join(model_id, "pytorch_model.bin"))
                
                checkpoint_dir = model_id
                
                if not has_weights:
                    # Find latest checkpoint
                    checkpoints = []
                    if os.path.isdir(model_id):
                        for item in os.listdir(model_id):
                            path = os.path.join(model_id, item)
                            if os.path.isdir(path) and item.startswith('checkpoint-'):
                                try:
                                    step = int(item.split('-')[-1])
                                    checkpoints.append((step, path))
                                except (ValueError, IndexError):
                                    continue
                    if checkpoints:
                        checkpoint_dir = sorted(checkpoints, key=lambda x: x[0], reverse=True)[0][1]
            
            # Load
            try:
                # 1. Load Config first to register custom architecture
                config = AutoConfig.from_pretrained(checkpoint_dir, local_files_only=True, trust_remote_code=True)
                
                # 2. Load Model
                model = AutoModelForImageClassification.from_pretrained(checkpoint_dir, config=config, local_files_only=True, trust_remote_code=True)
                
                # 2. Load Processor
                processor = None
                try:
                    # Try loading from root (model_id)
                    processor = AutoImageProcessor.from_pretrained(model_id, local_files_only=True, trust_remote_code=True)
                except (OSError, ValueError):
                    try:
                        # Try loading from checkpoint dir
                        processor = AutoImageProcessor.from_pretrained(checkpoint_dir, local_files_only=True, trust_remote_code=True)
                    except (OSError, ValueError):
                        # Fallback: If model has a backbone (custom_arcface), try to create timm transform
                        if hasattr(model, "backbone") and hasattr(model.config, "backbone"):
                            print(f"Processor config not found. Generating from timm backbone: {model.config.backbone}")
                            try:
                                import timm
                                data_config = timm.data.resolve_data_config({}, model=model.backbone)
                                transform = timm.data.create_transform(**data_config)
                                
                                # Wrapper to make timm transform behave like HF processor
                                class TimmProcessorWrapper:
                                    def __init__(self, transform):
                                        self.transform = transform
                                    def __call__(self, images, return_tensors="pt"):
                                        if isinstance(images, list):
                                            tensors = [self.transform(img) for img in images]
                                            pixel_values = torch.stack(tensors)
                                        else:
                                            pixel_values = self.transform(images).unsqueeze(0)
                                        return {"pixel_values": pixel_values}
                                
                                processor = TimmProcessorWrapper(transform)
                            except Exception as e:
                                print(f"Failed to create timm processor: {e}")

                if processor is None:
                    raise ValueError("Could not load preprocessor_config.json and failed to generate fallback.")

                model_type = "hf"
            except Exception as e:
                raise gr.Error(f"Error loading local model: {e}")

        else: 
            # Handle Hugging Face Hub
            try:
                processor = AutoImageProcessor.from_pretrained(model_id, trust_remote_code=True)
                model = AutoModelForImageClassification.from_pretrained(model_id, trust_remote_code=True)
                
                model_type = "hf"
            except Exception as e:
                raise gr.Error(f"Error loading from Hub: {e}")

    # --- CASE 2: Local .pth File (timm) ---
    elif source_type == "Local .pth":
        model_type = "timm"
        if not pth_file:
            raise gr.Error("Please provide path to .pth file.")
        if not pth_arch:
            raise gr.Error("Please specify the architecture name (e.g., resnet50).")
        
        try:
            import timm
        except ImportError:
            raise gr.Error("timm is required to load .pth files. Please install it via 'pip install timm'.")

        try:
            # 1. Load Weights
            try:
                state_dict = torch.load(pth_file, map_location=torch.device('cpu'), weights_only=True)
            except TypeError:
                # Fallback for older torch versions
                state_dict = torch.load(pth_file, map_location=torch.device('cpu'))
            
            if 'state_dict' in state_dict: state_dict = state_dict['state_dict']
            elif 'model' in state_dict: state_dict = state_dict['model']

            # 2. Infer num_classes
            num_classes = None
            for key in ['head.weight', 'fc.weight', 'classifier.weight']:
                if key in state_dict:
                    num_classes = state_dict[key].shape[0]
                    break

            # 3. Create Model
            clean_arch = pth_arch.replace("timm/", "") if pth_arch.startswith("timm/") else pth_arch
            kwargs = {'pretrained': False}
            if num_classes is not None: kwargs['num_classes'] = num_classes
            
            try:
                model = timm.create_model(clean_arch, **kwargs)
            except Exception:
                model = timm.create_model(pth_arch, **kwargs)
                
            model.load_state_dict(state_dict)
            model.eval()

            # 3. Preprocessing
            config = timm.data.resolve_data_config({}, model=model)
            processor = timm.data.create_transform(**config)
            
            # Load class names
            class_names = None
            if pth_classes:
                try:
                    if pth_classes.lower().endswith('.json'):
                        with open(pth_classes, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            if isinstance(data, list): class_names = data
                            elif isinstance(data, dict):
                                sorted_items = sorted(data.items(), key=lambda x: int(x[0]))
                                class_names = [v for k, v in sorted_items]
                    else:
                        with open(pth_classes, 'r', encoding='utf-8') as f:
                            class_names = [line.strip() for line in f if line.strip()]
                except Exception as e:
                    print(f"Warning: Failed to load class list: {e}")
            extra_data['class_names'] = class_names

        except Exception as e:
            raise gr.Error(f"Failed to load .pth file: {e}")

    return model, processor, model_type, extra_data

def get_gradcam(model, input_tensor, original_img):
    try:
        # Identify target layer (last Conv2d)
        target_layer = None
        # Iterate in reverse to find last Conv2d
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
                break
        
        if target_layer is None:
            return None

        gradients = []
        activations = []

        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])

        def forward_hook(module, input, output):
            activations.append(output)

        handle_b = target_layer.register_full_backward_hook(backward_hook)
        handle_f = target_layer.register_forward_hook(forward_hook)

        # Forward pass with gradients
        model.eval()
        model.zero_grad()
        
        # Handle input dict vs tensor
        if isinstance(input_tensor, torch.Tensor):
            output = model(input_tensor)
        else:
            try:
                output = model(**input_tensor)
            except TypeError:
                if "pixel_values" in input_tensor:
                    output = model(input_tensor["pixel_values"])
                else:
                    raise

        if hasattr(output, "logits"):
            logits = output.logits
        else:
            logits = output

        # Target class (max)
        pred_idx = logits.argmax(dim=1)
        score = logits[0, pred_idx]

        # Backward
        score.backward()

        handle_b.remove()
        handle_f.remove()

        if not gradients or not activations:
            return None

        grads = gradients[0].detach() # [1, C, H, W]
        acts = activations[0].detach() # [1, C, H, W]
        
        weights = torch.mean(grads, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * acts, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam_max = cam.max()
        if cam_max > 0:
            cam = cam / cam_max
        
        cam_np = cam.squeeze().cpu().numpy()
        
        # Resize to original image size
        w, h = original_img.size
        
        # Use matplotlib colormap
        cm = plt.get_cmap('jet')
        heatmap = cm(cam_np)[..., :3] # RGB, 0..1
        
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        heatmap_pil = Image.fromarray(heatmap_uint8).resize((w, h), resample=Image.BILINEAR)
        
        # Blend
        result = Image.blend(original_img.convert("RGB"), heatmap_pil, alpha=0.5)
        return result

    except Exception as e:
        print(f"GradCAM failed: {e}")
        return None

def classify_plant(source_type, local_path, hf_id, pth_file, pth_arch, pth_classes, input_image):
    if input_image is None:
        raise gr.Error("Please upload an image.")

    model, processor, model_type, extra_data = load_model_generic(source_type, local_path, hf_id, pth_file, pth_arch, pth_classes)
    
    predictions = {}
    heatmap = None

    if model_type == "hf":
        inputs = processor(images=input_image, return_tensors="pt")
        
        # 1. Prediction (No Grad)
        with torch.no_grad():
            if "pixel_values" in inputs:
                try:
                    outputs = model(**inputs)
                except TypeError:
                    outputs = model(inputs["pixel_values"])
            else:
                outputs = model(**inputs)

        if hasattr(outputs, "logits"):
            logits = outputs.logits
        else:
            logits = outputs

        probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
        top5_prob, top5_indices = torch.topk(probabilities, 5)
        predictions = {model.config.id2label[i.item()]: p.item() for i, p in zip(top5_indices, top5_prob)}
        
        # 2. Heatmap (With Grad)
        heatmap = get_gradcam(model, inputs, input_image)

    elif model_type == "timm":
        input_tensor = processor(input_image).unsqueeze(0)
        
        # 1. Prediction
        with torch.no_grad(): 
            output = model(input_tensor)
        
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top5_prob, top5_indices = torch.topk(probabilities, 5)
        
        class_names = extra_data.get('class_names')
        for i, p in zip(top5_indices, top5_prob):
            idx = i.item()
            label = class_names[idx] if class_names and idx < len(class_names) else f"Class {idx}"
            predictions[label] = p.item()
            
        # 2. Heatmap
        heatmap = get_gradcam(model, input_tensor, input_image)

    return predictions, heatmap

def predict_and_retrieve(source_type, local_path, hf_id, pth_file, pth_arch, pth_classes, herbarium_path, input_image):
    # 1. Classify
    try:
        predictions, heatmap = classify_plant(source_type, local_path, hf_id, pth_file, pth_arch, pth_classes, input_image)
    except Exception as e:
        raise gr.Error(f"Prediction failed: {e}")
    
    if not predictions:
        return None, None, None

    # Get top prediction
    top_class = max(predictions, key=predictions.get)
    
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
                
            except OSError:
                pass

    return predictions, herbarium_images, heatmap

# UI Construction
with gr.Blocks(theme=gr.themes.Monochrome(), css="footer {display: none !important}", title="Plant Species Identification") as demo:
    
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
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

            herbarium_dir = gr.Textbox(
                label="Herbarium dataset directory",
                placeholder="Path to folder containing class subfolders of herbarium images"
            )

            inf_input_image = gr.Image(type="pil", label="Upload field image")

        with gr.Column(scale=1):
            res_label = gr.Label(num_top_classes=5, label="Predictions")
            res_heatmap = gr.Image(label="Heatmap")
            res_gallery = gr.Gallery(label="Matching herbarium specimens", columns=3, height="auto")
            inf_button = gr.Button("Identify species", variant="primary")

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
        outputs=[res_label, res_gallery, res_heatmap]
    )

    def refresh_models():
        return update_model_choices()

    demo.load(fn=refresh_models, outputs=inf_model_path)

if __name__ == "__main__":
    demo.launch()
