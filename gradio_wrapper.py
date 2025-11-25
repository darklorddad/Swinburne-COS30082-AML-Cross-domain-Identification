import gradio as gr
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from safetensors.torch import load_file as safe_load_file
except ImportError:
    safe_load_file = None
import numpy as np
from PIL import Image
import os
import subprocess
import sys
import webbrowser
import time
import shutil
import requests
import random
import math
import matplotlib.pyplot as plt
import re
import json
import tempfile
from sklearn.manifold import TSNE

from utils import (
    util_plot_training_metrics,
    util_save_training_metrics
)



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
        ax.scatter(points[:, 0], points[:, 1], label=label, color=color, s=60, alpha=0.8)
    
    title = f"t-SNE Visualisation (MRR: {mrr_score:.4f})"
    if is_logits:
        title += "\n(Feature extraction failed; using logits)"
    ax.set_title(title)

    if len(unique_labels) <= 20:
        ax.legend()
    
    plt.tight_layout()
    return fig


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
                outputs = model(**inputs, output_hidden_states=True)
            except TypeError:
                # Model might not support output_hidden_states kwarg or **inputs
                if "pixel_values" in inputs:
                    outputs = model(inputs["pixel_values"])
                else:
                    outputs = model(**inputs)

            # Determine Logits
            if hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                logits = outputs

            # Determine Embeddings
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                batch_emb_numpy = outputs.pooler_output.cpu().numpy()
            elif hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                last_hidden = outputs.hidden_states[-1]
                if last_hidden.dim() == 4:
                    batch_emb_numpy = last_hidden.mean(dim=[2, 3]).cpu().numpy()
                elif last_hidden.dim() == 3:
                    batch_emb_numpy = last_hidden.mean(dim=1).cpu().numpy()
                else:
                    batch_emb_numpy = last_hidden.cpu().numpy()
            else:
                # LOUD FAILURE for feature extraction
                print("WARNING: Feature extraction failed! Model does not return hidden_states or pooler_output. Falling back to logits.")
                batch_emb_numpy = logits.cpu().numpy()
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
            for i in progress.tqdm(range(0, len(image_paths), batch_size), desc="Evaluating"):
                batch_paths = image_paths[i : i + batch_size]
                batch_labels = image_labels[i : i + batch_size]
                
                # Load images
                batch_images = []
                valid_indices = [] 
                for idx, p in enumerate(batch_paths):
                    try:
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


def launch_tensorboard(log_dir: str, venv_parent_dir: str):
    """Launches TensorBoard in a detached terminal using a specific venv."""
    if not log_dir or not os.path.isdir(log_dir):
        return "Error: Please provide a valid log directory."
    if not venv_parent_dir or not os.path.isdir(venv_parent_dir):
        return "Error: Please provide a valid parent directory for the venv."

    # Construct path to activate.bat (assuming Windows structure based on context)
    activate_script = os.path.join(venv_parent_dir, "venv", "Scripts", "activate.bat")
    if not os.path.exists(activate_script):
        return f"Error: Could not find activation script at: {activate_script}"

    # Construct the chained command
    # cmd /k keeps the window open. "call" is needed to run the bat file.
    full_command = f'cmd.exe /k call "{activate_script}" && tensorboard --logdir "{log_dir}"'
    tb_url = "http://localhost:6006"

    try:
        # Launch detached process
        subprocess.Popen(full_command, creationflags=subprocess.CREATE_NEW_CONSOLE)
        
        # Wait and check if it comes alive
        start_time = time.time()
        timeout = 20
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(tb_url, timeout=0.5)
                if response.status_code == 200:
                    webbrowser.open(tb_url)
                    return f"Success: TensorBoard launched in a new window.\nURL: {tb_url}"
            except (requests.ConnectionError, requests.Timeout):
                pass
            time.sleep(1)
            
        return f"Process launched, but failed to respond at {tb_url} within {timeout} seconds.\nPlease check the console window."

    except Exception as e:
        return f"Failed to launch TensorBoard: {e}"


def launch_autotrain_ui(autotrain_path: str):
    """Launches the AutoTrain Gradio UI in a separate detached process."""
    if not autotrain_path or not os.path.isdir(autotrain_path):
        return "Error: Please provide a valid path to the AutoTrain folder."

    autotrain_url = "http://localhost:7861"

    # Check if already running
    try:
        response = requests.get(autotrain_url, timeout=1)
        if response.status_code == 200:
            webbrowser.open(autotrain_url)
            return f"AutoTrain is already running. Opened {autotrain_url} in browser."
    except (requests.ConnectionError, requests.Timeout):
        pass

    module_parent_dir = os.path.dirname(autotrain_path)
    env = os.environ.copy()
    env['PYTHONPATH'] = f"{module_parent_dir}{os.pathsep}{env.get('PYTHONPATH', '')}"
    
    # Use absolute path for launch script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    launch_script = os.path.join(current_dir, 'launch_autotrain.py')
    if not os.path.exists(launch_script):
        launch_script = 'launch_autotrain.py'

    command = [sys.executable, launch_script]
    
    try:
        # Launch as a detached process
        if sys.platform == "win32":
            # CREATE_NEW_CONSOLE = 0x00000010
            subprocess.Popen(command, env=env, creationflags=subprocess.CREATE_NEW_CONSOLE)
        else:
            # start_new_session=True detaches the process on Unix
            subprocess.Popen(command, env=env, start_new_session=True)
        
        # Wait and check if it comes alive
        start_time = time.time()
        timeout = 30
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(autotrain_url, timeout=0.5)
                if response.status_code == 200:
                    webbrowser.open(autotrain_url)
                    return f"Success: AutoTrain UI launched in a new window.\nURL: {autotrain_url}"
            except (requests.ConnectionError, requests.Timeout):
                pass
            time.sleep(1)
            
        return f"Process launched, but failed to respond at {autotrain_url} within {timeout} seconds.\nPlease check the console window that opened."

    except Exception as e:
        return f"Failed to launch AutoTrain UI: {e}"

def show_model_charts(model_dir):
    """Finds trainer_state.json, returns metric plots."""
    if not model_dir:
        return (None,) * 11 + (gr.update(visible=False),)

    # The model_dir might be a checkpoint. trainer_state.json is usually in the parent.
    search_dir = model_dir
    if os.path.basename(search_dir).startswith('checkpoint-'):
        search_dir = os.path.dirname(search_dir)

    json_path = None
    # Look for trainer_state.json in the (potentially adjusted) search directory.
    for root, _, files in os.walk(search_dir):
        if 'trainer_state.json' in files:
            json_path = os.path.join(root, 'trainer_state.json')
            break

    if not json_path:
        print(f"trainer_state.json not found in '{search_dir}' or its subdirectories.")
        return (None,) * 11 + (gr.update(visible=False),)

    try:
        figures = util_plot_training_metrics(json_path)
        return (
            figures.get('Loss'), figures.get('Accuracy'), figures.get('Learning Rate'),
            figures.get('Gradient Norm'), figures.get('F1 Scores'), figures.get('Precision'),
            figures.get('Recall'), figures.get('Epoch'), figures.get('Eval Runtime'),
            figures.get('Eval Samples/sec'), figures.get('Eval Steps/sec'),
            gr.update(visible=True)
        )
    except Exception as e:
        print(f"Error generating plots for {json_path}: {e}")
        return (None,) * 11 + (gr.update(visible=False),)


def save_metrics(model_dir):
    if not model_dir:
        raise gr.Error("Please select a model.")
    
    # The model_dir might be a checkpoint. trainer_state.json is usually in the parent.
    search_dir = model_dir
    if os.path.basename(search_dir).startswith('checkpoint-'):
        search_dir = os.path.dirname(search_dir)

    # Define save directory inside the model root
    save_dir = os.path.join(search_dir, "Training-metrics")

    json_path = None
    for root, _, files in os.walk(search_dir):
        if 'trainer_state.json' in files:
            json_path = os.path.join(root, 'trainer_state.json')
            break

    if not json_path:
        raise gr.Error(f"trainer_state.json not found in '{search_dir}'")

    try:
        return util_save_training_metrics(json_path, save_dir)
    except Exception as e:
        raise gr.Error(f"Failed to save metrics: {e}")


def generate_manifest(directory_path: str, manifest_save_path: str, manifest_type: str):
    """Generates a manifest file listing all subdirectories and/or files."""
    if not directory_path or not os.path.isdir(directory_path):
        raise gr.Error("Please provide a valid directory path.")
    if not manifest_save_path:
        raise gr.Error("Please provide a manifest output path.")

    manifest_path = manifest_save_path
    if os.path.isdir(manifest_path):
        manifest_path = os.path.join(manifest_path, 'manifest.md')

    # Ensure the directory for the manifest file exists
    manifest_dir = os.path.dirname(manifest_path)
    if manifest_dir:
        os.makedirs(manifest_dir, exist_ok=True)

    try:
        manifest_items = []
        for root, dirs, files in os.walk(directory_path):
            # To ensure deterministic output, sort dirs and files
            dirs.sort()
            files.sort()
            
            # Add directories to manifest
            for d in dirs:
                full_path = os.path.join(root, d)
                relative_path = os.path.relpath(full_path, directory_path)
                manifest_items.append(relative_path.replace(os.sep, '/'))
            
            # Add files to manifest if requested
            if manifest_type == "Directories and files":
                for f in files:
                    full_path = os.path.join(root, f)
                    relative_path = os.path.relpath(full_path, directory_path)
                    if relative_path != '.':
                        manifest_items.append(relative_path.replace(os.sep, '/'))

        with open(manifest_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(sorted(manifest_items)))
        
        return f"Successfully generated manifest file at: {manifest_path}"
    except Exception as e:
        raise gr.Error(f"Failed to generate manifest file: {e}")


def to_snake_case(text):
    s = str(text).strip().lower()
    s = re.sub(r'[\s\-]+', '_', s)
    s = re.sub(r'[^a-z0-9_]', '', s)
    s = re.sub(r'_+', '_', s)
    return s.strip('_')


def clean_dataset_names(source_dir, destination_dir):
    if not destination_dir:
        raise gr.Error("Please provide a destination directory path.")
    if not source_dir or not os.path.isdir(source_dir):
        raise gr.Error("Please provide a valid source directory path.")

    try:
        copied_count = 0
        classes_processed = set()

        for root, dirs, files in os.walk(source_dir):
            if not files:
                continue
                
            # Skip the root folder itself if it has files (no class context)
            if os.path.abspath(root) == os.path.abspath(source_dir):
                continue

            original_class_name = os.path.basename(root)
            new_class_name = to_snake_case(original_class_name)
            if not new_class_name:
                new_class_name = "unknown"

            dest_class_path = os.path.join(destination_dir, new_class_name)
            os.makedirs(dest_class_path, exist_ok=True)
            classes_processed.add(new_class_name)

            for filename in files:
                name, ext = os.path.splitext(filename)
                if ext.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']:
                    new_filename = f"{to_snake_case(name)}{ext.lower()}"
                    shutil.copy2(os.path.join(root, filename), os.path.join(dest_class_path, new_filename))
                    copied_count += 1
        
        return f"Successfully cleaned dataset.\nSaved to: {destination_dir}\nClasses: {len(classes_processed)}\nFiles: {copied_count}"
    except Exception as e:
        raise gr.Error(f"Failed to clean dataset: {e}")


def split_dataset(source_dir, train_zip_path, val_zip_path, test_zip_path, train_manifest_path, val_manifest_path, test_manifest_path, split_type, train_ratio, val_ratio, test_ratio):
    """Splits a dataset into train, validation, and optional test sets."""
    
    def _generate_category_stats(class_dict, category_name):
        """Helper to generate summary statistics for a dictionary of classes."""
        report_lines = [f"\n## {category_name}"]
        if not class_dict:
            report_lines.append("None.")
            return report_lines

        num_classes = len(class_dict)
        total_items = sum(d['count'] for d in class_dict.values())
        
        report_lines.append(f"Total Classes: {num_classes}")
        report_lines.append(f"Total Items: {total_items}")

        if num_classes > 1:
            sorted_by_count = sorted(class_dict.items(), key=lambda item: item[1]['count'], reverse=True)
            most_common_name, most_common_data = sorted_by_count[0]
            least_common_name, least_common_data = sorted_by_count[-1]
            
            report_lines.append(f"Most Common: '{most_common_name}' with {most_common_data['count']} items")
            report_lines.append(f"Least Common: '{least_common_name}' with {least_common_data['count']} items")
            
            if least_common_data['count'] > 0:
                ratio = most_common_data['count'] / least_common_data['count']
                report_lines.append(f"Imbalance Ratio (Most/Least): {ratio:.1f}:1")
        elif num_classes == 1:
            class_name, data = list(class_dict.items())[0]
            report_lines.append(f"Only one class: '{class_name}' with {data['count']} items")

        return report_lines

    # --- 1. Input Validation ---
    if not source_dir or not os.path.isdir(source_dir): raise gr.Error("Please provide a valid source directory.")
    if not train_zip_path: raise gr.Error("Please provide a training set output path.")
    if not val_zip_path: raise gr.Error("Please provide a validation set output path.")
    if 'Test' in split_type and not test_zip_path: raise gr.Error("Please provide a test set output path.")
    if not train_manifest_path: raise gr.Error("Please provide a train manifest output path.")
    if not val_manifest_path: raise gr.Error("Please provide a validate manifest output path.")
    if 'Test' in split_type and not test_manifest_path: raise gr.Error("Please provide a test manifest output path.")

    output_paths = {'train': train_zip_path, 'validate': val_zip_path}
    manifest_paths = {'train': train_manifest_path, 'validate': val_manifest_path}
    if 'Test' in split_type:
        output_paths['test'] = test_zip_path
        manifest_paths['test'] = test_manifest_path

    for p in list(output_paths.values()) + list(manifest_paths.values()):
        if p:
            os.makedirs(os.path.dirname(p), exist_ok=True)

    train_r, val_r, test_r = train_ratio / 100.0, val_ratio / 100.0, test_ratio / 100.0
    total_ratio = train_r + val_r + (test_r if 'Test' in split_type else 0)
    if not math.isclose(total_ratio, 1.0): raise gr.Error(f"Ratios must sum to 100. Current sum is {total_ratio*100:.0f}.")

    # --- 2. Scan for classes and image files ---
    class_files = {}
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
    for root, dirs, files in os.walk(source_dir):
        if not dirs:  # It's a leaf directory
            class_name = os.path.basename(root)
            image_files = [os.path.join(root, f) for f in files if os.path.splitext(f)[1].lower() in image_extensions]
            if image_files: class_files[class_name] = image_files

    if not class_files: raise gr.Error("No leaf directories with images found in the source directory.")

    # --- 3. Split files for each class ---
    final_splits = {'train': {}, 'validate': {}}
    if 'Test' in split_type: final_splits['test'] = {}
    min_items_per_class, min_classes_per_set = 5, 2

    included_classes, skipped_classes = {}, {}

    for class_name, files in class_files.items():
        random.shuffle(files)
        n_total = len(files)
        
        # Calculate required items for each split, storing raw ratio-based values
        n_test, n_test_raw = 0, 0
        if 'Test' in split_type:
            n_test_raw = round(n_total * test_r)
            n_test = n_test_raw
            if 0 < n_test < min_items_per_class: n_test = min_items_per_class

        n_val_raw = round(n_total * val_r)
        n_val = n_val_raw
        if 0 < n_val < min_items_per_class: n_val = min_items_per_class
        
        n_train = n_total - n_test - n_val

        # "All-or-nothing" check
        is_test_valid = (test_r == 0) or (n_test >= min_items_per_class)
        is_val_valid = (val_r == 0) or (n_val >= min_items_per_class)
        is_train_valid = ((train_r == 0) or (n_train >= min_items_per_class)) and n_train >= 0

        if is_test_valid and is_val_valid and is_train_valid:
            included_classes[class_name] = {'count': n_total, 'splits': {'train': n_train, 'validate': n_val, 'test': n_test}}
            start_index = 0
            if n_test > 0:
                final_splits['test'][class_name] = files[start_index : start_index + n_test]
                start_index += n_test
            if n_val > 0:
                final_splits['validate'][class_name] = files[start_index : start_index + n_val]
                start_index += n_val
            if n_train > 0:
                final_splits['train'][class_name] = files[start_index:]
        else:
            reasons = []
            if n_train < 0:
                reasons.append(f"total items are too low (needs {n_test + n_val} for test/val, has {n_total})")
            else:
                if test_r > 0 and not is_test_valid:
                    reason = f"test set needs {min_items_per_class}"
                    if n_test_raw < min_items_per_class:
                        reason += f" (ratio gave {n_test_raw})"
                    reasons.append(reason)
                if val_r > 0 and not is_val_valid:
                    reason = f"validation set needs {min_items_per_class}"
                    if n_val_raw < min_items_per_class:
                        reason += f" (ratio gave {n_val_raw})"
                    reasons.append(reason)
                if train_r > 0 and not is_train_valid:
                    reasons.append(f"train set needs {min_items_per_class} (only {n_train} left)")
            
            reason_str = "insufficient items: " + ", ".join(reasons) if reasons else "an unknown reason"
            skipped_classes[class_name] = {'count': n_total, 'reason': reason_str}

    # --- 4. Post-split validation ---
    for set_name, classes in final_splits.items():
        if 0 < len(classes) < min_classes_per_set:
            raise gr.Error(f"Could not create '{set_name}' split. It would have only {len(classes)} class(es), but the minimum is {min_classes_per_set}.")

    # --- 5. Create zip archives ---
    temp_parent_dir = os.path.join(os.path.dirname(train_zip_path), f"temp_split_{int(time.time())}")
    os.makedirs(temp_parent_dir, exist_ok=True)
    created_zips = []

    try:
        for set_name, classes in final_splits.items():
            if not classes:
                continue

            set_dir = os.path.join(temp_parent_dir, set_name)
            os.makedirs(set_dir, exist_ok=True)
            manifest_files = []

            # --- Default file copying for validation/test/train ---
            for class_name, files_to_copy in classes.items():
                class_dir = os.path.join(set_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)
                for f in files_to_copy:
                    shutil.copy2(f, class_dir)
                    file_name = os.path.basename(f)
                    manifest_files.append(f"{class_name}/{file_name}".replace(os.sep, '/'))

            # --- Build manifest content with summary ---
            manifest_content = [f"# {set_name.capitalize()} Set Manifest"]

            set_class_counts = {name: data['splits'][set_name] for name, data in included_classes.items() if data['splits'].get(set_name, 0) > 0}
            
            manifest_content.append("\n## Set Summary")
            if not set_class_counts:
                manifest_content.append("No classes were included in this set.")
            else:
                num_included = len(set_class_counts)
                manifest_content.append(f"Total classes: {num_included}")
                manifest_content.append(f"Total items: {sum(set_class_counts.values())}")
                manifest_content.append("\n### Class Counts")
                for class_name, count in sorted(set_class_counts.items()):
                    manifest_content.append(f"- {class_name}: {count} items")

            manifest_content.extend(_generate_category_stats(included_classes, "All Included Classes (from source)"))
            manifest_content.extend(_generate_category_stats(skipped_classes, "All Skipped Classes (from source)"))

            if skipped_classes:
                manifest_content.append("\n## Skipped Class Details")
                manifest_content.append("Classes are skipped if they don't have enough items for the required splits.")
                for name, data in sorted(skipped_classes.items()):
                    manifest_content.append(f"- {name} ({data['count']} items): Skipped because {data['reason']}")

            manifest_content.append("\n## File List")
            manifest_content.extend(sorted(manifest_files))

            with open(os.path.join(set_dir, 'manifest.md'), 'w', encoding='utf-8') as f:
                f.write('\n'.join(manifest_content))

            external_manifest_path = manifest_paths[set_name]
            with open(external_manifest_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(manifest_content))

            zip_path = output_paths[set_name]
            zip_path_base = os.path.splitext(zip_path)[0]
            archive_path = shutil.make_archive(zip_path_base, 'zip', set_dir)
            created_zips.append(archive_path)

    finally:
        if os.path.exists(temp_parent_dir): shutil.rmtree(temp_parent_dir)

    if not created_zips: return "No datasets were created. Check source data and split ratios."
    return f"Successfully created dataset splits: {', '.join(created_zips)}"


def check_dataset_balance(source_dir: str, save_files: bool, chart_save_path: str, manifest_save_path: str):
    """Checks the balance of a dataset by counting files in leaf directories."""
    if not source_dir or not os.path.isdir(source_dir):
        raise gr.Error("Please provide a valid source directory.")

    try:
        class_counts = {}
        for root, dirs, files in os.walk(source_dir):
            if not dirs:  # Leaf directory
                if os.path.abspath(root) != os.path.abspath(source_dir):
                    class_name = os.path.basename(root)
                    class_counts[class_name] = len(files)

        if not class_counts:
            return None, "No leaf directories with items found in the source directory."

        # Sort by count (descending) for the report
        sorted_class_counts = sorted(class_counts.items(), key=lambda item: item[1], reverse=True)
        total_items = sum(class_counts.values())

        # Create plot with classes sorted alphabetically
        plot_classes, plot_counts = zip(*sorted(class_counts.items()))
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.bar(plot_classes, plot_counts)
        ax.set_title('Dataset Class Distribution')
        ax.set_xlabel('Class')
        ax.set_ylabel('Number of Items')
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        status_messages = ["Successfully generated balance chart and statistics."]
        
        # Generate report content
        report_lines = ["\n## Dataset Statistics"]
        if total_items > 0:
            report_lines.append(f"Total Classes: {len(sorted_class_counts)}")
            report_lines.append(f"Total Items: {total_items}")
            
            # Add most and least common classes
            most_common_name, most_common_count = sorted_class_counts[0]
            least_common_name, least_common_count = sorted_class_counts[-1]
            most_common_ratio = (most_common_count / total_items) * 100
            least_common_ratio = (least_common_count / total_items) * 100
            report_lines.append(f"Most Common: '{most_common_name}' with {most_common_count} items ({most_common_ratio:.2f}%)")
            report_lines.append(f"Least Common: '{least_common_name}' with {least_common_count} items ({least_common_ratio:.2f}%)")
            if least_common_count > 0:
                imbalance_ratio = most_common_count / least_common_count
                report_lines.append(f"Imbalance Ratio (Most/Least): {imbalance_ratio:.1f}:1")

            report_lines.append("\n### Class Counts and Ratios")
            for class_name, count in sorted_class_counts:
                ratio = (count / total_items) * 100
                report_lines.append(f"- {class_name}: {count} items ({ratio:.2f}%)")
        else:
            report_lines.append("No items found.")
        
        status_messages.extend(report_lines)

        # Save chart and manifest if requested
        if save_files:
            if chart_save_path:
                try:
                    chart_dir = os.path.dirname(chart_save_path)
                    if chart_dir:
                        os.makedirs(chart_dir, exist_ok=True)
                    fig.savefig(chart_save_path)
                    status_messages.append(f"Chart saved to: {chart_save_path}")
                except Exception as e:
                    status_messages.append(f"Warning: Could not save chart: {e}")

            if manifest_save_path:
                try:
                    manifest_dir = os.path.dirname(manifest_save_path)
                    if manifest_dir:
                        os.makedirs(manifest_dir, exist_ok=True)

                    manifest_content = ["# Dataset Balance Manifest"] + report_lines

                    with open(manifest_save_path, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(manifest_content))
                    status_messages.append(f"Manifest saved to: {manifest_save_path}")
                except Exception as e:
                    status_messages.append(f"Warning: Could not save manifest: {e}")

        return fig, '\n'.join(status_messages)

    except Exception as e:
        raise gr.Error(f"Failed to check dataset balance: {e}")


def check_dataset_splittability(source_dir, split_type, train_ratio, val_ratio, test_ratio):
    """Simulates dataset splitting and provides a detailed report on the expected outcome."""
    
    def _generate_category_stats(class_dict, category_name):
        """Helper to generate summary statistics for a dictionary of classes."""
        report_lines = [f"\n## {category_name} Classes"]
        if not class_dict:
            report_lines.append("None.")
            return report_lines

        num_classes = len(class_dict)
        total_items = sum(d['count'] for d in class_dict.values())
        
        report_lines.append(f"Total Classes: {num_classes}")
        report_lines.append(f"Total Items: {total_items}")

        if num_classes > 1:
            sorted_by_count = sorted(class_dict.items(), key=lambda item: item[1]['count'], reverse=True)
            most_common_name, most_common_data = sorted_by_count[0]
            least_common_name, least_common_data = sorted_by_count[-1]
            
            report_lines.append(f"Most Common: '{most_common_name}' with {most_common_data['count']} items")
            report_lines.append(f"Least Common: '{least_common_name}' with {least_common_data['count']} items")
            
            if least_common_data['count'] > 0:
                ratio = most_common_data['count'] / least_common_data['count']
                report_lines.append(f"Imbalance Ratio (Most/Least): {ratio:.1f}:1")
        elif num_classes == 1:
            class_name, data = list(class_dict.items())[0]
            report_lines.append(f"Only one class: '{class_name}' with {data['count']} items")

        return report_lines

    # --- 1. Input Validation ---
    if not source_dir or not os.path.isdir(source_dir):
        raise gr.Error("Please provide a valid source directory.")
    
    train_r, val_r, test_r = train_ratio / 100.0, val_ratio / 100.0, test_ratio / 100.0
    total_ratio = train_r + val_r + (test_r if 'Test' in split_type else 0)
    if not math.isclose(total_ratio, 1.0):
        raise gr.Error(f"Ratios must sum to 100. Current sum is {total_ratio*100:.0f}.")

    # --- 2. Scan for classes and file counts ---
    class_counts = {}
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
    for root, dirs, files in os.walk(source_dir):
        if not dirs:  # It's a leaf directory
            class_name = os.path.basename(root)
            image_files = [f for f in files if os.path.splitext(f)[1].lower() in image_extensions]
            if image_files:
                class_counts[class_name] = len(image_files)

    if not class_counts:
        return "No leaf directories with images found in the source directory."

    # --- 3. Simulate splitting to categorise classes ---
    included_classes, skipped_classes = {}, {}
    min_items_per_class, min_classes_per_set = 5, 2

    for class_name, n_total in class_counts.items():
        # Calculate required items for each split, storing raw ratio-based values
        n_test, n_test_raw = 0, 0
        if 'Test' in split_type:
            n_test_raw = round(n_total * test_r)
            n_test = n_test_raw
            if 0 < n_test < min_items_per_class: n_test = min_items_per_class

        n_val_raw = round(n_total * val_r)
        n_val = n_val_raw
        if 0 < n_val < min_items_per_class: n_val = min_items_per_class
        
        n_train = n_total - n_test - n_val

        # "All-or-nothing" check
        is_test_valid = (test_r == 0) or (n_test >= min_items_per_class)
        is_val_valid = (val_r == 0) or (n_val >= min_items_per_class)
        is_train_valid = ((train_r == 0) or (n_train >= min_items_per_class)) and n_train >= 0

        if is_test_valid and is_val_valid and is_train_valid:
            included_classes[class_name] = {'count': n_total, 'splits': {'train': n_train, 'validate': n_val, 'test': n_test}}
        else:
            reasons = []
            if n_train < 0:
                reasons.append(f"total items are too low (needs {n_test + n_val} for test/val, has {n_total})")
            else:
                if test_r > 0 and not is_test_valid:
                    reason = f"test set needs {min_items_per_class}"
                    if n_test_raw < min_items_per_class:
                        reason += f" (ratio gave {n_test_raw})"
                    reasons.append(reason)
                if val_r > 0 and not is_val_valid:
                    reason = f"validation set needs {min_items_per_class}"
                    if n_val_raw < min_items_per_class:
                        reason += f" (ratio gave {n_val_raw})"
                    reasons.append(reason)
                if train_r > 0 and not is_train_valid:
                    reasons.append(f"train set needs {min_items_per_class} (only {n_train} left)")
            
            reason_str = "insufficient items: " + ", ".join(reasons) if reasons else "an unknown reason"
            skipped_classes[class_name] = {'count': n_total, 'reason': reason_str}

    # --- 4. Generate report ---
    report_lines = ["# Splittability Report"]

    # --- Post-split validation and outcome summary ---
    final_outcome_messages = []
    set_names = ['train', 'validate']
    if 'Test' in split_type: set_names.append('test')

    for set_name in set_names:
        set_class_counts = {name: data['splits'][set_name] for name, data in included_classes.items() if data['splits'][set_name] > 0}
        num_included = len(set_class_counts)
        
        if 0 < num_included < min_classes_per_set:
            final_outcome_messages.append(f"The '{set_name}' set would only contain {num_included} class(es), but the minimum required is {min_classes_per_set}.")

    report_lines.append("\n## Final Outcome")
    if final_outcome_messages:
        report_lines.append("**FAILURE**: The dataset cannot be split with these settings. The `split_dataset` function would raise an error.")
        report_lines.append("\n**Reasons:**")
        for msg in final_outcome_messages:
            report_lines.append(f"- {msg}")
    else:
        report_lines.append("**SUCCESS**: The dataset can be split with the current settings.")

    # --- Detailed Breakdown ---
    report_lines.extend(_generate_category_stats(included_classes, "Included"))
    report_lines.extend(_generate_category_stats(skipped_classes, "Skipped"))

    if included_classes:
        report_lines.append("\n## Included Set Breakdown")
        # set_names is already defined

        for set_name in set_names:
            report_lines.append(f"\n### {set_name.capitalize()} Set")
            set_class_counts = {name: data['splits'][set_name] for name, data in included_classes.items() if data['splits'][set_name] > 0}
            
            if not set_class_counts:
                report_lines.append("No classes will be included in this set.")
                continue

            num_included = len(set_class_counts)
            report_lines.append(f"Total classes: {num_included}")
            report_lines.append(f"Total items: {sum(set_class_counts.values())}")
            report_lines.append("\n#### Class Counts")
            for class_name, count in sorted(set_class_counts.items()):
                report_lines.append(f"- {class_name}: {count} items")

    if skipped_classes:
        report_lines.append("\n## Skipped Class Details")
        report_lines.append("Classes are skipped if they don't have enough items for the required splits.")
        for name, data in sorted(skipped_classes.items()):
            report_lines.append(f"- {name} ({data['count']} items): Skipped because it {data['reason']}")

    return '\n'.join(report_lines)


def get_model_choices(task="inference"):
    """
    Finds model directories. 
    task: 'inference', 'evaluation' (requires weights) or 'metrics' (requires trainer_state.json)
    """
    choices = []
    try:
        app_dir = os.path.dirname(os.path.abspath(__file__))
        model_parent_dirs = [d for d in os.listdir(app_dir) if os.path.isdir(os.path.join(app_dir, d)) and d.startswith('Model-')]

        for model_dir_name in model_parent_dirs:
            model_dir_path = os.path.join(app_dir, model_dir_name)
            
            if task == "metrics":
                # Check for trainer_state.json recursively
                has_logs = False
                for root, _, files in os.walk(model_dir_path):
                    if "trainer_state.json" in files:
                        has_logs = True
                        break
                
                if has_logs:
                    choices.append((model_dir_name, model_dir_path))
            
            else:
                # Inference/Evaluation: Check for weights
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

def update_model_choices(task="inference"):
    """Refreshes the list of available models."""
    choices = get_model_choices(task)
    return gr.update(choices=choices, filterable=False)
