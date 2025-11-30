import os
import zipfile
import gradio as gr
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoConfig
import transformers.modeling_outputs

# Patch ImageClassifierOutput to accept pooler_output which some custom models erroneously pass
# We modify the __init__ in-place so that any reference to the class uses the patched version.
# We store the original init on the class to avoid recursion issues during reloads.
if not hasattr(transformers.modeling_outputs.ImageClassifierOutput, "_original_init_backup"):
    transformers.modeling_outputs.ImageClassifierOutput._original_init_backup = transformers.modeling_outputs.ImageClassifierOutput.__init__

    def _new_init(self, *args, **kwargs):
        kwargs.pop("pooler_output", None)
        transformers.modeling_outputs.ImageClassifierOutput._original_init_backup(self, *args, **kwargs)
    
    transformers.modeling_outputs.ImageClassifierOutput.__init__ = _new_init

from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import re
import tempfile
from sklearn.manifold import TSNE

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
    if source_type in ["Local", "Hugging Face hub"]:
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

            # --- Auto-Adapt Head for Sequential Layers (Fix for fc.0.weight errors) ---
            # Detect if state_dict has a sequential head (e.g., fc.0.weight) while model has a single layer
            head_name = None
            if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear): head_name = 'fc'
            elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Linear): head_name = 'classifier'
            elif hasattr(model, 'head') and isinstance(model.head, nn.Linear): head_name = 'head'

            if head_name:
                # Check if state_dict contains keys like "fc.0.weight"
                seq_keys = [k for k in state_dict.keys() if k.startswith(f"{head_name}.0.weight")]
                if seq_keys and f"{head_name}.weight" not in state_dict:
                    print(f"Detected Sequential head in .pth for '{head_name}'. Adapting model structure...")
                    
                    # Extract dimensions from state_dict
                    # Layer 0 (First Linear)
                    w0 = state_dict[f"{head_name}.0.weight"]
                    in_features = w0.shape[1]
                    hidden_dim = w0.shape[0]
                    
                    # Layer 3 (Final Linear) - assuming standard fastai/fine-tuning structure
                    # If fc.3 exists, we assume structure: Linear -> ReLU -> Dropout -> Linear
                    if f"{head_name}.3.weight" in state_dict:
                        w3 = state_dict[f"{head_name}.3.weight"]
                        out_features = w3.shape[0]
                        
                        new_head = nn.Sequential(
                            nn.Linear(in_features, hidden_dim),
                            nn.ReLU(inplace=True),
                            nn.Dropout(0.5), # Standard default, though exact rate isn't in weights
                            nn.Linear(hidden_dim, out_features)
                        )
                        setattr(model, head_name, new_head)
                    
                    # Fallback: If only fc.0 exists (unlikely for this specific error, but possible)
                    elif f"{head_name}.0.weight" in state_dict and f"{head_name}.1.weight" not in state_dict:
                         # Just a wrapped linear?
                         setattr(model, head_name, nn.Sequential(nn.Linear(in_features, hidden_dim)))

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
            except (TypeError, ValueError):
                try:
                    if "pixel_values" in input_tensor:
                        output = model(input_tensor["pixel_values"])
                    else:
                        raise
                except (TypeError, ValueError):
                    if "pixel_values" in input_tensor:
                        output = model(input_tensor["pixel_values"], return_dict=False)
                    else:
                        output = model(**input_tensor, return_dict=False)

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
            try:
                if "pixel_values" in inputs:
                    try:
                        outputs = model(**inputs)
                    except TypeError:
                        outputs = model(inputs["pixel_values"])
                else:
                    outputs = model(**inputs)
            except TypeError:
                # Fallback for models with ImageClassifierOutput init issues
                if "pixel_values" in inputs:
                    outputs = model(inputs["pixel_values"], return_dict=False)
                else:
                    outputs = model(**inputs, return_dict=False)

        if hasattr(outputs, "logits"):
            logits = outputs.logits
        elif isinstance(outputs, tuple):
            logits = outputs[0]
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

def get_placeholder_plot():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, "Evaluation in progress...", ha='center', va='center', fontsize=12, color='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    return fig

def plot_tsne(embeddings, true_labels, mrr_score, is_logits=False, perplexity=30):
    if len(embeddings) < 2:
        return None
        
    embeddings_np = np.array(embeddings)
    n_samples = embeddings_np.shape[0]
    safe_perplexity = min(perplexity, n_samples - 1)
    
    tsne = TSNE(n_components=2, perplexity=safe_perplexity, random_state=42, init='pca', learning_rate='auto')
    tsne_results = tsne.fit_transform(embeddings_np)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    unique_labels = list(set(true_labels))
    cmap = plt.get_cmap('tab10' if len(unique_labels) <= 10 else 'viridis')
    
    for i, label in enumerate(unique_labels):
        indices = [j for j, l in enumerate(true_labels) if l == label]
        points = tsne_results[indices]
        color = cmap(i / len(unique_labels))
        ax.scatter(points[:, 0], points[:, 1], label=label, color=color, s=10, alpha=0.8)
    
    title = "t-SNE Visualisation"
    if is_logits:
        title += "\n(Feature extraction failed; using logits)"
    ax.set_title(title)

    if len(unique_labels) <= 20:
        ax.legend()
    
    plt.tight_layout()
    return fig

def plot_metrics(mrr, top1, top5):
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = ['MRR', 'Top-1 Acc', 'Top-5 Acc']
    values = [mrr, top1, top5]
    bars = ax.bar(metrics, values, color=['#3498db', '#2ecc71', '#9b59b6'])
    
    ax.set_ylim(0, 1.0)
    ax.set_title('Evaluation Metrics')
    ax.set_ylabel('Score')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

def extract_features_and_logits(model, processor, batch_images, device, model_type):
    batch_emb_numpy = None
    logits = None
    fallback_to_logits = False

    with torch.no_grad():
        if model_type == "hf":
            inputs = processor(images=batch_images, return_tensors="pt")
            if hasattr(inputs, "to"):
                inputs = inputs.to(device)
            else:
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            # Try to get hidden states
            try:
                outputs = model(**inputs, output_hidden_states=True, return_dict=True)
            except (TypeError, ValueError):
                # Try return_dict=False to bypass ImageClassifierOutput init errors (e.g. pooler_output issue)
                try:
                    outputs = model(**inputs, output_hidden_states=True, return_dict=False)
                except (TypeError, ValueError):
                    # Model might not support output_hidden_states kwarg or **inputs
                    try:
                        if "pixel_values" in inputs:
                            # Try passing control args explicitly with pixel_values as kwargs
                            outputs = model(pixel_values=inputs["pixel_values"], output_hidden_states=True, return_dict=True)
                        else:
                            outputs = model(**inputs)
                    except (TypeError, ValueError) as e:
                        print(f"Feature extraction attempt failed: {e}")
                        try:
                            if "pixel_values" in inputs:
                                outputs = model(inputs["pixel_values"])
                            else:
                                outputs = model(**inputs)
                        except (TypeError, ValueError):
                            if "pixel_values" in inputs:
                                outputs = model(inputs["pixel_values"], return_dict=False)
                            else:
                                outputs = model(**inputs, return_dict=False)

            # Determine Logits
            if hasattr(outputs, "logits"):
                logits = outputs.logits
            elif isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            # Determine Embeddings
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                batch_emb_numpy = outputs.pooler_output.cpu().numpy()
            elif isinstance(outputs, dict) and 'pooler_output' in outputs and outputs['pooler_output'] is not None:
                batch_emb_numpy = outputs['pooler_output'].cpu().numpy()
            elif hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                last_hidden = outputs.hidden_states[-1]
                if last_hidden.dim() == 4:
                    batch_emb_numpy = last_hidden.mean(dim=[2, 3]).cpu().numpy()
                elif last_hidden.dim() == 3:
                    batch_emb_numpy = last_hidden.mean(dim=1).cpu().numpy()
                else:
                    batch_emb_numpy = last_hidden.cpu().numpy()
            elif isinstance(outputs, dict) and 'hidden_states' in outputs and outputs['hidden_states']:
                last_hidden = outputs['hidden_states'][-1]
                if last_hidden.dim() == 4:
                    batch_emb_numpy = last_hidden.mean(dim=[2, 3]).cpu().numpy()
                elif last_hidden.dim() == 3:
                    batch_emb_numpy = last_hidden.mean(dim=1).cpu().numpy()
                else:
                    batch_emb_numpy = last_hidden.cpu().numpy()
            elif isinstance(outputs, tuple) and len(outputs) > 1:
                # Handle tuple output (logits, hidden_states, ...)
                possible_hidden = outputs[1]
                if isinstance(possible_hidden, (tuple, list)) and len(possible_hidden) > 0 and isinstance(possible_hidden[0], torch.Tensor):
                    last_hidden = possible_hidden[-1]
                    if last_hidden.dim() == 4:
                        batch_emb_numpy = last_hidden.mean(dim=[2, 3]).cpu().numpy()
                    elif last_hidden.dim() == 3:
                        batch_emb_numpy = last_hidden.mean(dim=1).cpu().numpy()
                    else:
                        batch_emb_numpy = last_hidden.cpu().numpy()
                else:
                    # Fallback if tuple structure isn't as expected
                    print("WARNING: Feature extraction failed (tuple output)! Falling back to logits.")
                    batch_emb_numpy = logits.cpu().numpy() if isinstance(logits, torch.Tensor) else logits
                    fallback_to_logits = True
            else:
                # Try accessing backbone directly (Custom ArcFace fallback)
                if hasattr(model, "backbone"):
                    try:
                        if "pixel_values" in inputs:
                            pv = inputs["pixel_values"]
                        else:
                            # Try to find tensor in inputs values
                            pv = next(v for v in inputs.values() if isinstance(v, torch.Tensor))
                        
                        # Forward pass through backbone only
                        features = model.backbone(pv)
                        
                        # Apply simple pooling
                        if features.dim() == 4:
                            batch_emb_numpy = features.mean(dim=[2, 3]).cpu().numpy()
                        elif features.dim() == 3:
                            batch_emb_numpy = features.mean(dim=1).cpu().numpy()
                        else:
                            batch_emb_numpy = features.cpu().numpy()
                            
                    except Exception as e:
                        print(f"Backbone fallback failed: {e}")
                        batch_emb_numpy = None

                if batch_emb_numpy is None:
                    # LOUD FAILURE for feature extraction
                    print("WARNING: Feature extraction failed! Model does not return hidden_states or pooler_output. Falling back to logits.")
                    batch_emb_numpy = logits.cpu().numpy() if isinstance(logits, torch.Tensor) else logits
                    fallback_to_logits = True

        elif model_type == "timm":
            tensors = [processor(img) for img in batch_images]
            input_tensor = torch.stack(tensors).to(device)
            try:
                features = model.forward_features(input_tensor)
                if features.dim() == 4:
                    batch_emb_numpy = features.mean(dim=[2, 3]).cpu().numpy()
                elif features.dim() == 3:
                    batch_emb_numpy = features.mean(dim=1).cpu().numpy()
                else:
                    batch_emb_numpy = features.cpu().numpy()
                
                if hasattr(model, 'forward_head'):
                    logits = model.forward_head(features)
                else:
                    logits = model(input_tensor)
            except Exception:
                logits = model(input_tensor)
                batch_emb_numpy = logits.cpu().numpy()
                fallback_to_logits = True
                
    return batch_emb_numpy, logits, fallback_to_logits

def evaluate_test_set(source_type, local_path, hf_id, pth_file, pth_arch, pth_classes, test_dir, batch_size=32, perplexity=30, eval_mode="Standard", reference_dir=None):
    if not test_dir or not os.path.exists(test_dir):
        raise gr.Error("Please provide a valid test directory.")

    model, processor, model_type, extra_data = load_model_generic(source_type, local_path, hf_id, pth_file, pth_arch, pth_classes)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # --- PROTOTYPE RETRIEVAL SETUP ---
    prototypes_tensor = None
    
    if eval_mode == "Prototype retrieval":
        if not reference_dir or not os.path.exists(reference_dir):
            raise gr.Error("Please provide a valid reference directory for prototype retrieval.")
        
        gr.Info("Computing prototypes from reference set...")
        
        # Scan reference directory
        ref_paths = []
        ref_labels = []
        for root, dirs, files in os.walk(reference_dir):
            if os.path.abspath(root) == os.path.abspath(reference_dir): continue
            class_name = os.path.basename(root)
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')):
                    ref_paths.append(os.path.join(root, f))
                    ref_labels.append(class_name)
        
        if not ref_paths:
            raise gr.Error("No images found in reference directory.")

        # Compute embeddings for reference set
        ref_embeddings = []
        ref_labels_processed = []
        
        for i in range(0, len(ref_paths), batch_size):
            batch_p = ref_paths[i : i + batch_size]
            batch_l = ref_labels[i : i + batch_size]
            
            batch_imgs = []
            valid_l = []
            for idx, p in enumerate(batch_p):
                try:
                    img = Image.open(p).convert("RGB")
                    batch_imgs.append(img)
                    valid_l.append(batch_l[idx])
                except Exception:
                    pass
            
            if not batch_imgs: continue
            
            feats, _, _ = extract_features_and_logits(model, processor, batch_imgs, device, model_type)
            if feats is not None:
                ref_embeddings.extend(feats)
                ref_labels_processed.extend(valid_l)
        
        if not ref_embeddings:
            raise gr.Error("Failed to extract features from reference set.")

        # Compute Prototypes (Mean Embedding per Class)
        ref_embeddings = np.array(ref_embeddings)
        unique_classes = sorted(list(set(ref_labels_processed)))
        
        prototypes = []
        for cls in unique_classes:
            indices = [i for i, l in enumerate(ref_labels_processed) if l == cls]
            class_embs = ref_embeddings[indices]
            # Mean and Normalize
            proto = np.mean(class_embs, axis=0)
            proto = proto / (np.linalg.norm(proto) + 1e-9)
            prototypes.append(proto)
            
        prototypes_tensor = torch.tensor(np.array(prototypes), device=device, dtype=torch.float)
        
        # Override label mapping for evaluation
        id2label = {i: name for i, name in enumerate(unique_classes)}
        label2id = {name: i for i, name in enumerate(unique_classes)}
        
    else:
        # Standard Mode: Use Model's Classes
        id2label = {}
        label2id = {}
        
        if model_type == "hf" or model_type == "custom_arcface":
            id2label = model.config.id2label
            label2id = {v: k for k, v in id2label.items()}
        elif model_type == "timm":
            class_names = extra_data.get('class_names')
            if class_names:
                id2label = {i: name for i, name in enumerate(class_names)}
                label2id = {name: i for i, name in enumerate(class_names)}
    
    # Scan directory structure for images
    # Expected structure: test_dir/class_name/image.jpg
    image_paths = []
    image_labels = []
    
    # Helper for fuzzy matching (snake_case normalization)
    def normalize_name(name):
        s = str(name).strip().lower()
        s = re.sub(r'[\s\-]+', '_', s)
        s = re.sub(r'[^a-z0-9_]', '', s)
        return s.strip('_')

    # Create a normalized map of model labels
    # label2id keys are the class names the model knows
    norm_model_labels = {}
    for k in label2id.keys():
        # 1. Full normalized match
        norm_model_labels[normalize_name(k)] = k
        # 2. If format is "ID; Name", match against Name part
        if ';' in k:
            parts = k.split(';', 1)
            if len(parts) == 2:
                norm_model_labels[normalize_name(parts[1])] = k
    
    found_folders = []
    skipped_folders = []
    
    is_zip = os.path.isfile(test_dir) and test_dir.lower().endswith('.zip')

    if is_zip:
        try:
            zf = zipfile.ZipFile(test_dir, 'r')
            all_files = zf.namelist()
            
            for filepath in all_files:
                if filepath.endswith('/'): continue
                if not filepath.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')):
                    continue
                
                # Extract folder name (class name)
                parts = filepath.strip('/').split('/')
                if len(parts) < 2: continue
                
                folder_name = parts[-2]
                
                matched_label = None
                if folder_name in label2id:
                    matched_label = folder_name
                elif normalize_name(folder_name) in norm_model_labels:
                    matched_label = norm_model_labels[normalize_name(folder_name)]
                
                if matched_label:
                    image_paths.append(filepath)
                    image_labels.append(matched_label)
                    if folder_name not in found_folders: found_folders.append(folder_name)
                else:
                    if folder_name not in skipped_folders: skipped_folders.append(folder_name)
            zf.close()
        except Exception as e:
            raise gr.Error(f"Failed to read ZIP file: {e}")
    else:
        for root, dirs, files in os.walk(test_dir):
            if os.path.abspath(root) == os.path.abspath(test_dir):
                continue # Skip root folder
                
            folder_name = os.path.basename(root)
            found_folders.append(folder_name)
            
            # Try exact match
            matched_label = None
            if folder_name in label2id:
                matched_label = folder_name
            # Try normalized match
            elif normalize_name(folder_name) in norm_model_labels:
                matched_label = norm_model_labels[normalize_name(folder_name)]
            
            if not matched_label:
                skipped_folders.append(folder_name)
                continue
                
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')):
                    image_paths.append(os.path.join(root, f))
                    image_labels.append(matched_label) # Use the model's label, not the folder name

    if not image_paths:
        # Generate a helpful error message
        model_classes_sample = list(label2id.keys())[:5]
        folders_sample = found_folders[:5]
        msg = (
            f"No valid images found. \n"
            f"Checked {len(found_folders)} subfolders in '{test_dir}'.\n"
            f"Sample folders found: {folders_sample}\n"
            f"Sample model classes: {model_classes_sample}\n"
            f"Ensure folder names match model class names (case-insensitive, snake_case handled)."
        )
        raise gr.Error(msg)

    true_labels = []
    ranks = []
    top1_correct = 0
    top5_correct = 0
    total_processed = 0
    fallback_to_logits = False
    
    # Use a temporary file to store embeddings to avoid OOM on large datasets
    embeddings_file = tempfile.NamedTemporaryFile(delete=False, suffix='.npy')
    embeddings_path = embeddings_file.name
    embeddings_file.close()
    
    progress = gr.Progress()
    
    try:
        with open(embeddings_path, 'wb') as f_emb:
            # Batch processing loop
            zip_obj = zipfile.ZipFile(test_dir, 'r') if is_zip else None
            try:
                for i in progress.tqdm(range(0, len(image_paths), batch_size), desc="Evaluating"):
                    batch_paths = image_paths[i : i + batch_size]
                    batch_labels = image_labels[i : i + batch_size]
                    
                    # Load images
                    batch_images = []
                    valid_indices = [] 
                    for idx, p in enumerate(batch_paths):
                        try:
                            if is_zip:
                                with zip_obj.open(p) as file_in_zip:
                                    img = Image.open(file_in_zip).convert("RGB")
                                    batch_images.append(img)
                            else:
                                img = Image.open(p).convert("RGB")
                                batch_images.append(img)
                            valid_indices.append(idx)
                        except Exception:
                            pass
                    
                    if not batch_images: continue

                    # Inference
                    batch_emb_numpy, logits, fallback = extract_features_and_logits(model, processor, batch_images, device, model_type)
                    
                    if fallback:
                        fallback_to_logits = True

                    # Save embeddings to disk
                    if batch_emb_numpy is not None:
                        np.save(f_emb, batch_emb_numpy)

                    # Determine Logits for Metrics
                    if eval_mode == "Prototype retrieval" and prototypes_tensor is not None and batch_emb_numpy is not None:
                        # Cosine Similarity: (B, D) @ (C, D).T -> (B, C)
                        # Ensure embeddings are normalized
                        feats_t = torch.tensor(batch_emb_numpy, device=device)
                        feats_t = torch.nn.functional.normalize(feats_t, dim=1)
                        
                        # Prototypes are already normalized
                        sim_logits = torch.matmul(feats_t, prototypes_tensor.t())
                        
                        # Use similarity as logits
                        final_logits = sim_logits
                    else:
                        # Standard Mode
                        if isinstance(logits, torch.Tensor):
                            final_logits = logits
                        else:
                            final_logits = torch.tensor(logits, device=device)

                    # Metrics Calculation
                    # For cosine similarity, higher is better, so softmax works (or just argsort directly)
                    # Softmax is monotonic, so argsort order is preserved.
                    probs = torch.nn.functional.softmax(final_logits, dim=1)
                    sorted_indices = torch.argsort(probs, descending=True, dim=1)
                    
                    for j, valid_idx in enumerate(valid_indices):
                        gt_label = batch_labels[valid_idx]
                        gt_id = label2id[gt_label]
                        
                        # Rank (MRR)
                        try:
                            rank = (sorted_indices[j] == gt_id).nonzero(as_tuple=True)[0].item() + 1
                            ranks.append(1.0 / rank)
                        except Exception:
                            ranks.append(0)

                        # Top-1
                        if sorted_indices[j][0] == gt_id:
                            top1_correct += 1
                            
                        # Top-5
                        if gt_id in sorted_indices[j][:5]:
                            top5_correct += 1
                            
                        true_labels.append(gt_label)
                        total_processed += 1
            finally:
                if zip_obj:
                    zip_obj.close()
        
        if total_processed == 0:
            raise gr.Error("No valid labeled images found.")

        # Calculate final metrics
        mrr = np.mean(ranks)
        top1_acc = top1_correct / total_processed
        top5_acc = top5_correct / total_processed
        
        # Load embeddings back for t-SNE (handling memory limit)
        embeddings_list = []
        try:
            with open(embeddings_path, 'rb') as f_emb:
                while True:
                    try:
                        batch = np.load(f_emb)
                        embeddings_list.extend(batch)
                    except (ValueError, EOFError, OSError):
                        break
        except (FileNotFoundError, OSError):
            pass
            
        # If too many embeddings, downsample for t-SNE to avoid OOM/Freeze
        MAX_TSNE_SAMPLES = 3000
        tsne_embeddings = embeddings_list
        tsne_labels = true_labels
        
        if len(embeddings_list) > MAX_TSNE_SAMPLES:
            indices = np.random.choice(len(embeddings_list), MAX_TSNE_SAMPLES, replace=False)
            tsne_embeddings = [embeddings_list[i] for i in indices]
            tsne_labels = [true_labels[i] for i in indices]
        
        # Generate t-SNE Plot
        tsne_fig = plot_tsne(tsne_embeddings, tsne_labels, mrr, is_logits=fallback_to_logits, perplexity=perplexity)
        
        # Generate Metrics Plot
        metrics_fig = plot_metrics(mrr, top1_acc, top5_acc)

        # Prepare results dict for export
        results_dict = {
            "mrr": mrr,
            "top1": top1_acc,
            "top5": top5_acc,
            "total": total_processed,
            "embeddings": embeddings_list, 
            "true_labels": true_labels,
            "skipped_folders": skipped_folders,
            "fallback_to_logits": fallback_to_logits,
            "perplexity": perplexity
        }

        return tsne_fig, metrics_fig, results_dict

    finally:
        # Cleanup temp file
        if os.path.exists(embeddings_path):
            os.remove(embeddings_path)

def save_evaluation_results(results_dict, output_dir):
    if not results_dict:
        return "No evaluation results to save. Please run evaluation first."
    if not output_dir:
        return "Please specify an output directory."
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 1. Save Single JSON with details
        # Convert numpy arrays in embeddings to lists for JSON serialization
        embeddings_list = [e.tolist() if isinstance(e, np.ndarray) else e for e in results_dict.get("embeddings", [])]
        
        data_to_save = {
            "mrr": results_dict.get("mrr"),
            "top1_accuracy": results_dict.get("top1"),
            "top5_accuracy": results_dict.get("top5"),
            "total_images": results_dict.get("total"),
            "skipped_folders": results_dict.get("skipped_folders", []),
            "true_labels": results_dict.get("true_labels"),
            "embeddings": embeddings_list 
        }
        
        with open(os.path.join(output_dir, "evaluation_results.json"), "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, indent=4)
            
        # 2. Save Plots as Images
        # t-SNE
        embeddings = results_dict.get("embeddings")
        true_labels = results_dict.get("true_labels")
        mrr = results_dict.get("mrr")
        fallback = results_dict.get("fallback_to_logits", False)
        perplexity = results_dict.get("perplexity", 30)
        
        if embeddings and true_labels:
            fig = plot_tsne(embeddings, true_labels, mrr, is_logits=fallback, perplexity=perplexity)
            if fig:
                fig.savefig(os.path.join(output_dir, "tsne_plot.png"))
                plt.close(fig)
        
        # Metrics Plot
        mrr = results_dict.get("mrr")
        top1 = results_dict.get("top1")
        top5 = results_dict.get("top5")
        if mrr is not None:
            fig = plot_metrics(mrr, top1, top5)
            fig.savefig(os.path.join(output_dir, "metrics_plot.png"))
            plt.close(fig)
                
        return f"Successfully saved evaluation results to {output_dir}"
    except Exception as e:
        return f"Failed to save results: {e}"

def predict_and_retrieve(source_type, local_path, hf_id, pth_file, pth_arch, pth_classes, input_image):
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
    
    app_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check for ZIP first, then Folder
    dataset_zip = os.path.join(app_dir, "Dataset.zip")
    dataset_dir = os.path.join(app_dir, "Dataset")
    
    if not os.path.exists(dataset_zip) and not os.path.exists(dataset_dir):
        parent_dir = os.path.dirname(app_dir)
        dataset_zip = os.path.join(parent_dir, "Dataset.zip")
        dataset_dir = os.path.join(parent_dir, "Dataset")

    # Logic for ZIP
    if os.path.isfile(dataset_zip):
        try:
            with zipfile.ZipFile(dataset_zip, 'r') as zf:
                all_files = zf.namelist()
                
                def normalize(n): return str(n).lower().replace(' ', '_').replace('-', '_')
                norm_top = normalize(top_class)
                
                candidates = []
                
                for f in all_files:
                    if not f.lower().endswith(('.jpg', '.jpeg', '.png')): continue
                    if 'herbarium' not in f.lower(): continue
                    
                    parts = f.lower().split('/')
                    if norm_top in [normalize(p) for p in parts]:
                        candidates.append(f)
                
                if candidates:
                    random.shuffle(candidates)
                    selected_files = candidates[:6]
                    
                    for zip_path in selected_files:
                        with zf.open(zip_path) as img_file:
                            img = Image.open(img_file).convert("RGB")
                            img_copy = img.copy() 
                            herbarium_images.append(img_copy)
                            
        except Exception as e:
            print(f"Error reading gallery from zip: {e}")

    # Logic for Folder
    elif os.path.isdir(dataset_dir):
        dataset_path = os.path.join(dataset_dir, "Split-set")
        if os.path.isdir(dataset_path):
            target_dir = None
            
            possible_path = os.path.join(dataset_path, top_class)
            if os.path.isdir(possible_path):
                target_dir = possible_path
            else:
                def normalize(n): return str(n).lower().replace(' ', '_').replace('-', '_')
                norm_top = normalize(top_class)
                try:
                    for d in os.listdir(dataset_path):
                        full_d = os.path.join(dataset_path, d)
                        if os.path.isdir(full_d):
                            if normalize(d) == norm_top:
                                target_dir = full_d
                                break
                except OSError:
                    pass
            
            if target_dir:
                try:
                    files = [os.path.join(target_dir, f) for f in os.listdir(target_dir) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'))
                             and 'herbarium' in f.lower()]
                    random.shuffle(files)
                    herbarium_images = files[:6]
                except OSError:
                    pass

    return predictions, herbarium_images, heatmap

# UI Construction
app_dir = os.path.dirname(os.path.abspath(__file__))
dataset_zip = os.path.join(app_dir, "Dataset.zip")
dataset_dir = os.path.join(app_dir, "Dataset")

if not os.path.exists(dataset_zip) and not os.path.exists(dataset_dir):
    parent_dir = os.path.dirname(app_dir)
    dataset_zip = os.path.join(parent_dir, "Dataset.zip")
    dataset_dir = os.path.join(parent_dir, "Dataset")

if os.path.isfile(dataset_zip):
    default_test_dir = dataset_zip
    default_ref_dir = dataset_zip
else:
    default_test_dir = os.path.join(dataset_dir, "Test-set")
    default_ref_dir = os.path.join(dataset_dir, "Split-set")

with gr.Blocks(theme=gr.themes.Monochrome(), css="footer {display: none !important}", title="Plant Species Identification") as demo:
    
    with gr.Tab("Inference"):
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    inf_source = gr.Radio(
                        choices=["Local", "Hugging Face hub", "Local .pth"],
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

                inf_button = gr.Button("Identify species", variant="primary")
                inf_input_image = gr.Image(type="pil", label="Upload field image")

            with gr.Column(scale=1):
                res_label = gr.Label(num_top_classes=5, label="Predictions")
                res_gallery = gr.Gallery(label="Matching herbarium specimens", columns=3, height="auto")
                res_heatmap = gr.Image(label="Heatmap")

        # Event Handlers for Inference
        def update_inf_inputs(source):
            return (
                gr.update(visible=(source == "Local")),
                gr.update(visible=(source == "Hugging Face hub")),
                gr.update(visible=(source == "Local .pth"))
            )

        inf_source.change(
            fn=update_inf_inputs,
            inputs=[inf_source],
            outputs=[inf_model_path, inf_hf_id, inf_pth_group]
        )

        inf_button.click(
            fn=predict_and_retrieve,
            inputs=[inf_source, inf_model_path, inf_hf_id, inf_pth_file, inf_pth_arch, inf_pth_classes, inf_input_image],
            outputs=[res_label, res_gallery, res_heatmap]
        )

    with gr.Tab("Evaluation"):
        # 1. Model Selection
        with gr.Row():
            with gr.Column():
                with gr.Group():
                    eval_source = gr.Radio(
                        choices=["Local", "Hugging Face hub", "Local .pth"],
                        value="Local",
                        label="Model source"
                    )
                
                    eval_model_path = gr.Dropdown(
                        label="Select local model", 
                        choices=[], 
                        value=None, 
                        filterable=True,
                        visible=True,
                        allow_custom_value=True
                    )
                    eval_hf_id = gr.Textbox(
                        label="Hugging Face model ID", 
                        visible=False
                    )
                    with gr.Column(visible=False) as eval_pth_group:
                        eval_pth_file = gr.Textbox(label="Path to .pth file")
                        eval_pth_classes = gr.Textbox(label="Path to class list (txt/json)")
                        eval_pth_arch = gr.Textbox(
                            label="Architecture name (timm)"
                        )

        # 2. Test Set & Run
        with gr.Column(visible=True) as eval_run_container:
            with gr.Accordion("Settings", open=False):
                eval_mode = gr.Radio(["Standard", "Prototype retrieval"], label="Evaluation mode", value="Standard")
                eval_batch_size = gr.Slider(minimum=1, maximum=128, value=32, step=1, label="Batch size")
                eval_perplexity = gr.Slider(minimum=2, maximum=100, value=30, step=1, label="t-SNE perplexity")
            eval_button = gr.Button("Run evaluation", variant="primary")

        # 4. Results (Hidden until run)
        eval_results_state = gr.State()
        
        with gr.Column(visible=False) as eval_results_container:
            with gr.Row():
                eval_plot_tsne = gr.Plot(label="t-SNE visualisation", value=get_placeholder_plot())
                eval_plot_metrics = gr.Plot(label="Metrics", value=get_placeholder_plot())

        # Logic for Evaluation
        def update_eval_inputs(source):
            is_local = (source == "Local")
            is_hf = (source == "Hugging Face hub")
            is_pth = (source == "Local .pth")
            
            return (
                gr.update(visible=is_local),
                gr.update(visible=is_hf),
                gr.update(visible=is_pth)
            )

        eval_source.change(
            fn=update_eval_inputs,
            inputs=[eval_source],
            outputs=[eval_model_path, eval_hf_id, eval_pth_group]
        )

        eval_button.click(
            fn=lambda: gr.update(visible=True),
            outputs=[eval_results_container]
        ).then(
            fn=lambda src, path, hf, pth, arch, cls, bs, perp, mode: evaluate_test_set(
                src, path, hf, pth, arch, cls, 
                default_test_dir, 
                bs, perp, mode, 
                default_ref_dir
            ),
            inputs=[eval_source, eval_model_path, eval_hf_id, eval_pth_file, eval_pth_arch, eval_pth_classes, eval_batch_size, eval_perplexity, eval_mode],
            outputs=[eval_plot_tsne, eval_plot_metrics, eval_results_state]
        )

    def refresh_models():
        u = update_model_choices()
        return u, u

    demo.load(fn=refresh_models, outputs=[inf_model_path, eval_model_path])

if __name__ == "__main__":
    demo.launch()
