import os
import shutil
import re
import gradio as gr
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def get_latest_checkpoint(model_path):
    if not os.path.exists(model_path):
        return None
        
    if os.path.basename(model_path).startswith('checkpoint-'):
        return model_path
        
    checkpoints = []
    for item in os.listdir(model_path):
        path = os.path.join(model_path, item)
        if os.path.isdir(path) and item.startswith('checkpoint-'):
            try:
                step = int(item.split('-')[-1])
                checkpoints.append((step, path))
            except (ValueError, IndexError):
                continue

    if checkpoints:
        return sorted(checkpoints, key=lambda x: x[0], reverse=True)[0][1]
    
    # If no checkpoints found, check if the dir itself has config.json
    if os.path.exists(os.path.join(model_path, 'config.json')):
        return model_path
        
    return None

def evaluate_model(model_path, test_dir):
    if not model_path:
        raise gr.Error("Please select a model.")
    if not test_dir or not os.path.exists(test_dir):
        raise gr.Error("Please provide a valid test directory.")

    checkpoint_path = get_latest_checkpoint(model_path)
    if not checkpoint_path:
        raise gr.Error(f"No valid model checkpoint found in {model_path}")

    try:
        processor = AutoImageProcessor.from_pretrained(checkpoint_path, local_files_only=True)
        model = AutoModelForImageClassification.from_pretrained(checkpoint_path, local_files_only=True)
    except Exception as e:
        raise gr.Error(f"Failed to load model: {e}")

    model.eval()
    
    # Prepare labels
    id2label = model.config.id2label
    label2id = {v: k for k, v in id2label.items()}
    
    # We need to match filenames to labels. 
    # Filenames are like: {label}_{original_name}.jpg
    # We sort labels by length to match longest prefix first.
    sorted_labels = sorted(label2id.keys(), key=len, reverse=True)
    
    true_labels = []
    embeddings = []
    ranks = []
    
    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        raise gr.Error("No images found in test directory.")

    progress = gr.Progress()
    
    with torch.no_grad():
        for i, filename in enumerate(progress.tqdm(image_files, desc="Processing images")):
            # 1. Identify Ground Truth
            gt_label = None
            for label in sorted_labels:
                if filename.startswith(label + "_"):
                    gt_label = label
                    break
            
            if not gt_label:
                # Skip files where we can't identify the class
                continue
                
            gt_id = label2id[gt_label]
            
            # 2. Load and Process Image
            img_path = os.path.join(test_dir, filename)
            try:
                image = Image.open(img_path).convert("RGB")
                inputs = processor(images=image, return_tensors="pt")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
                
            # 3. Inference
            outputs = model(**inputs, output_hidden_states=True)
            logits = outputs.logits[0]
            
            # Use the last hidden state for embeddings
            if outputs.hidden_states:
                # Global Average Pooling on the last hidden state
                last_hidden = outputs.hidden_states[-1][0] 
                if last_hidden.dim() == 2: # [seq_len, hidden]
                    emb = last_hidden.mean(dim=0)
                elif last_hidden.dim() == 3: # [channels, h, w]
                    emb = last_hidden.mean(dim=[1, 2])
                else:
                    emb = last_hidden
                embeddings.append(emb.numpy())
            else:
                embeddings.append(logits.numpy())

            true_labels.append(gt_label)
            
            # 4. Calculate Rank for MRR
            sorted_indices = torch.argsort(logits, descending=True)
            rank = (sorted_indices == gt_id).nonzero(as_tuple=True)[0].item() + 1
            ranks.append(1.0 / rank)

    if not ranks:
        return "No valid labeled images found for evaluation.", None

    # MRR
    mrr_score = np.mean(ranks)
    result_text = f"Mean Reciprocal Rank (MRR): {mrr_score:.4f}\nProcessed {len(ranks)} images."

    # t-SNE
    if len(embeddings) < 2:
        return result_text + "\nNot enough data for t-SNE.", None
        
    embeddings_np = np.array(embeddings)
    n_samples = embeddings_np.shape[0]
    perplexity = min(30, n_samples - 1)
    
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init='pca', learning_rate='auto')
    tsne_results = tsne.fit_transform(embeddings_np)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    unique_labels = list(set(true_labels))
    # Simple colormap
    cmap = plt.get_cmap('tab10' if len(unique_labels) <= 10 else 'viridis')
    
    for i, label in enumerate(unique_labels):
        indices = [j for j, l in enumerate(true_labels) if l == label]
        points = tsne_results[indices]
        color = cmap(i / len(unique_labels))
        ax.scatter(points[:, 0], points[:, 1], label=label, color=color, s=60, alpha=0.8)
    
    ax.set_title(f"t-SNE Visualisation (MRR: {mrr_score:.4f})")
    if len(unique_labels) <= 20:
        ax.legend()
    else:
        result_text += "\n(Legend hidden due to high number of classes)"
    
    plt.tight_layout()
    
    return result_text, fig

def plot_tsne(embeddings, true_labels, mrr_score):
    if len(embeddings) < 2:
        return None
        
    embeddings_np = np.array(embeddings)
    n_samples = embeddings_np.shape[0]
    perplexity = min(30, n_samples - 1)
    
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init='pca', learning_rate='auto')
    tsne_results = tsne.fit_transform(embeddings_np)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    unique_labels = list(set(true_labels))
    cmap = plt.get_cmap('tab10' if len(unique_labels) <= 10 else 'viridis')
    
    for i, label in enumerate(unique_labels):
        indices = [j for j, l in enumerate(true_labels) if l == label]
        points = tsne_results[indices]
        color = cmap(i / len(unique_labels))
        ax.scatter(points[:, 0], points[:, 1], label=label, color=color, s=60, alpha=0.8)
    
    ax.set_title(f"t-SNE Visualisation (MRR: {mrr_score:.4f})")
    if len(unique_labels) <= 20:
        ax.legend()
    
    plt.tight_layout()
    return fig

def sort_test_dataset(test_dir, destination_dir, groundtruth_path, species_list_path):
    if not os.path.exists(test_dir):
        raise gr.Error(f"Test directory not found: {test_dir}")
    if not destination_dir:
        raise gr.Error("Please provide a destination directory.")
    if not os.path.exists(groundtruth_path):
        raise gr.Error(f"Groundtruth file not found: {groundtruth_path}")
    if not os.path.exists(species_list_path):
        raise gr.Error(f"Species list file not found: {species_list_path}")

    # Load species list
    class_to_species = {}
    try:
        with open(species_list_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                parts = line.split(';')
                if len(parts) >= 2:
                    class_id = parts[0].strip()
                    species_name = parts[1].strip()
                    # Sanitize
                    s = str(species_name).strip().lower()
                    s = re.sub(r'[\s\-]+', '_', s)
                    s = re.sub(r'[^a-z0-9_]', '', s)
                    safe_species_name = re.sub(r'_+', '_', s).strip('_')
                    class_to_species[class_id] = safe_species_name
    except Exception as e:
        raise gr.Error(f"Error reading species list: {e}")

    # Load groundtruth
    id_to_class = {}
    try:
        with open(groundtruth_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                parts = line.split()
                if len(parts) >= 2:
                    # format: test/1745.jpg 105951
                    filename_part = parts[0]
                    class_id = parts[1]
                    
                    # Extract ID from filename part (e.g. test/1745.jpg -> 1745)
                    basename = os.path.basename(filename_part)
                    img_id = os.path.splitext(basename)[0]
                    id_to_class[img_id] = class_id
    except Exception as e:
        raise gr.Error(f"Error reading groundtruth: {e}")
    
    copied_count = 0
    errors = []
    
    os.makedirs(destination_dir, exist_ok=True)

    # Iterate over files in test directory
    for filename in os.listdir(test_dir):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        # Try to extract ID from filename
        name_without_ext = os.path.splitext(filename)[0]
        
        # If it contains underscores, assume the ID is the last part (handling previously renamed files if any)
        if '_' in name_without_ext:
            parts = name_without_ext.split('_')
            img_id = parts[-1]
        else:
            img_id = name_without_ext
            
        if img_id not in id_to_class:
            continue
            
        class_id = id_to_class[img_id]
        species_name = class_to_species.get(class_id, "unknown_species")
        
        # Create class directory
        class_dir = os.path.join(destination_dir, species_name)
        os.makedirs(class_dir, exist_ok=True)
        
        source_path = os.path.join(test_dir, filename)
        dest_path = os.path.join(class_dir, filename)
        
        try:
            shutil.copy2(source_path, dest_path)
            copied_count += 1
        except OSError as e:
            errors.append(f"Error copying {filename}: {e}")

    result_msg = f"Finished. Sorted {copied_count} files into {destination_dir}."
    if errors:
        result_msg += "\nErrors:\n" + "\n".join(errors[:10])
        if len(errors) > 10:
            result_msg += f"\n...and {len(errors)-10} more errors."
            
    return result_msg

def custom_sort_dataset(source_dir, destination_dir, species_list_path, pairs_list_path):
    """Sorts dataset into class folders named by species, renaming images with metadata."""
    if not destination_dir:
        raise gr.Error("Please provide a destination directory path.")
    if not source_dir or not os.path.isdir(source_dir):
        raise gr.Error("Please provide a valid source directory path.")

    # Load species mapping
    id_to_name = {}
    if species_list_path and os.path.isfile(species_list_path):
        try:
            with open(species_list_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(';')
                    if len(parts) >= 2:
                        id_val = parts[0].strip()
                        name_val = parts[1].strip()
                        # Sanitize for filesystem
                        safe_name = "".join([c if c.isalnum() or c in " .-_()" else "_" for c in name_val])
                        id_to_name[id_val] = safe_name
        except Exception as e:
            print(f"Warning: Failed to parse species list: {e}")

    # Load pairs list
    ids_with_pairs = set()
    if pairs_list_path and os.path.isfile(pairs_list_path):
        try:
            with open(pairs_list_path, 'r', encoding='utf-8') as f:
                for line in f:
                    ids_with_pairs.add(line.strip())
        except Exception as e:
            print(f"Warning: Failed to parse pairs list: {e}")

    try:
        copied_files_count = 0
        created_folders = set()

        for root, dirs, files in os.walk(source_dir):
            if not files:
                continue
            
            # Determine ID from folder name (assuming leaf dir is ID)
            class_id = os.path.basename(root)
            
            # Use mapped name if available, else ID
            class_name = id_to_name.get(class_id, class_id)
            
            # Determine Type (herbarium/photo) from path
            path_parts = root.replace('\\', '/').split('/')
            image_type = "unknown"
            if 'herbarium' in path_parts:
                image_type = "herbarium"
            elif 'photo' in path_parts:
                image_type = "photo"
            
            # Determine Pair Status
            is_pair = "pair" if class_id in ids_with_pairs else "no_pair"
            
            dest_class_dir = os.path.join(destination_dir, class_name)
            os.makedirs(dest_class_dir, exist_ok=True)
            created_folders.add(class_name)

            for filename in files:
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')):
                    src_file_path = os.path.join(root, filename)
                    
                    # Construct new filename
                    name_part = class_name.replace(" ", "_")
                    new_filename = f"{name_part}_{image_type}_{is_pair}_{filename}"
                    
                    dest_file_path = os.path.join(dest_class_dir, new_filename)
                    
                    shutil.copy2(src_file_path, dest_file_path)
                    copied_files_count += 1

        return f"Successfully sorted dataset at: {destination_dir}\nCreated {len(created_folders)} class folders.\nCopied {copied_files_count} files."

    except Exception as e:
        raise gr.Error(f"Failed to sort dataset: {e}")

def rename_test_images_func(test_dir, groundtruth_path, species_list_path):
    if not os.path.exists(test_dir):
        raise gr.Error(f"Test directory not found: {test_dir}")

    if not os.path.exists(groundtruth_path):
        raise gr.Error(f"Groundtruth file not found: {groundtruth_path}")

    if not os.path.exists(species_list_path):
        raise gr.Error(f"Species list file not found: {species_list_path}")

    # Load species list
    class_to_species = {}
    try:
        with open(species_list_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                parts = line.split(';')
                if len(parts) >= 2:
                    class_id = parts[0].strip()
                    species_name = parts[1].strip()
                    class_to_species[class_id] = species_name
    except Exception as e:
        raise gr.Error(f"Error reading species list: {e}")

    # Load groundtruth
    id_to_class = {}
    try:
        with open(groundtruth_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                parts = line.split()
                if len(parts) >= 2:
                    # format: test/1745.jpg 105951
                    filename_part = parts[0]
                    class_id = parts[1]
                    
                    # Extract ID from filename part (e.g. test/1745.jpg -> 1745)
                    basename = os.path.basename(filename_part)
                    img_id = os.path.splitext(basename)[0]
                    id_to_class[img_id] = class_id
    except Exception as e:
        raise gr.Error(f"Error reading groundtruth: {e}")
    
    renamed_count = 0
    errors = []
    
    # Iterate over files in test directory
    for filename in os.listdir(test_dir):
        if not filename.lower().endswith(".jpg"):
            continue
            
        # Try to extract ID from filename
        # It could be "1000.jpg" or "test_unknown_no_pair_1000.jpg"
        name_without_ext = os.path.splitext(filename)[0]
        
        # If it contains underscores, assume the ID is the last part
        if '_' in name_without_ext:
            parts = name_without_ext.split('_')
            img_id = parts[-1]
        else:
            img_id = name_without_ext
            
        if img_id not in id_to_class:
            continue
            
        class_id = id_to_class[img_id]
        species_name = class_to_species.get(class_id, "Unknown")
        
        # Sanitize species name for filename (snake_case)
        s = str(species_name).strip().lower()
        s = re.sub(r'[\s\-]+', '_', s)
        s = re.sub(r'[^a-z0-9_]', '', s)
        safe_species_name = re.sub(r'_+', '_', s).strip('_')
        
        # Check if already renamed
        if filename.startswith(f"{safe_species_name}_"):
             continue

        # Construct new filename
        # Format: {SpeciesName}_{OriginalName}
        new_filename = f"{safe_species_name}_{filename}"
        
        source_path = os.path.join(test_dir, filename)
        dest_path = os.path.join(test_dir, new_filename)
        
        try:
            os.rename(source_path, dest_path)
            renamed_count += 1
        except OSError as e:
            errors.append(f"Error renaming {filename}: {e}")

    result_msg = f"Finished. Renamed {renamed_count} files."
    if errors:
        result_msg += "\nErrors:\n" + "\n".join(errors[:10])
        if len(errors) > 10:
            result_msg += f"\n...and {len(errors)-10} more errors."
            
    return result_msg
