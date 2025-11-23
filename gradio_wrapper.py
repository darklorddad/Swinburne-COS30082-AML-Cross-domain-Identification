import gradio as gr
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from PIL import Image
import os
import subprocess
import sys
import webbrowser
import signal
import time
import shutil
import requests
import random
import zipfile
import math
import matplotlib.pyplot as plt
import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors
import queue
import threading
import re

from utils import (
    util_plot_training_metrics
)

AUTOTRAIN_PROCESS = None


def _enqueue_output(stream, queue_obj):
    """Reads from a stream and puts lines into a queue."""
    try:
        for line in iter(stream.readline, ''):
            queue_obj.put(line)
    finally:
        stream.close()


def classify_plant(model_path: str, input_image: Image.Image) -> dict:
    if not model_path:
        raise gr.Error("Please select a model directory.")

    model_dir = model_path
    if os.path.isfile(model_path):
        model_dir = os.path.dirname(model_path)

    # If model_dir is a checkpoint, the actual model root is its parent.
    if os.path.basename(model_dir).startswith('checkpoint-'):
        checkpoint_dir = model_dir
        model_dir = os.path.dirname(model_dir)
    else:
        # It's a model directory. Find the latest checkpoint.
        checkpoint_dir = model_dir  # Default to model_dir if no checkpoints
        checkpoints = []
        if os.path.isdir(model_dir):
            for item in os.listdir(model_dir):
                path = os.path.join(model_dir, item)
                if os.path.isdir(path) and item.startswith('checkpoint-'):
                    try:
                        step = int(item.split('-')[-1])
                        checkpoints.append((step, path))
                    except (ValueError, IndexError):
                        continue  # Not a valid checkpoint folder name

        if checkpoints:
            latest_checkpoint_path = sorted(checkpoints, key=lambda x: x[0], reverse=True)[0][1]
            checkpoint_dir = latest_checkpoint_path
            print(f"Found latest checkpoint for '{model_dir}': '{checkpoint_dir}'")

    try:
        image_processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True)
        model = AutoModelForImageClassification.from_pretrained(checkpoint_dir, local_files_only=True)
    except Exception as e:
        raise gr.Error(f"Error loading model from {checkpoint_dir}. Check path and files. Original error: {e}")
    inputs = image_processor(images=input_image, return_tensors="pt")
    with torch.no_grad(): outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    top5_prob, top5_indices = torch.topk(probabilities, 5)
    return {model.config.id2label[i.item()]: p.item() for i, p in zip(top5_indices, top5_prob)}


def launch_autotrain_ui(autotrain_path: str):
    """Launches the AutoTrain Gradio UI, streams its output, and opens it in a new browser tab."""
    if not autotrain_path or not os.path.isdir(autotrain_path):
        yield "Error: Please provide a valid path to the AutoTrain folder.", gr.update(visible=True), gr.update(visible=False)
        return

    global AUTOTRAIN_PROCESS
    if AUTOTRAIN_PROCESS and AUTOTRAIN_PROCESS.poll() is None:
        yield "AutoTrain UI is already running.", gr.update(interactive=False), gr.update(visible=True)
        return

    module_parent_dir = os.path.dirname(autotrain_path)
    env = os.environ.copy()
    env['PYTHONPATH'] = f"{module_parent_dir}{os.pathsep}{env.get('PYTHONPATH', '')}"
    command = [sys.executable, os.path.join('launch_autotrain.py')]
    autotrain_url = "http://localhost:7861"
    
    log_output = "Launching AutoTrain UI...\n"
    yield log_output, gr.update(interactive=False), gr.update(visible=True)

    try:
        startupinfo = None
        if sys.platform == "win32":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        popen_kwargs = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "bufsize": 1,
            "env": env,
            "text": True,
        }

        if sys.platform == "win32":
            popen_kwargs["startupinfo"] = startupinfo
        else:
            popen_kwargs["preexec_fn"] = os.setsid

        AUTOTRAIN_PROCESS = subprocess.Popen(command, **popen_kwargs)
        
        output_queue = queue.Queue()
        reader_thread = threading.Thread(target=_enqueue_output, args=(AUTOTRAIN_PROCESS.stdout, output_queue))
        reader_thread.daemon = True
        reader_thread.start()
        print("Started thread to read AutoTrain output.")

        start_time = time.time()
        timeout = 30
        server_ready = False
        
        while AUTOTRAIN_PROCESS.poll() is None:
            try:
                while True:
                    line = output_queue.get_nowait()
                    print(line, end='')
                    log_output += line
                    yield log_output, gr.update(interactive=False), gr.update(visible=True)
            except queue.Empty:
                pass

            if not server_ready:
                try:
                    response = requests.get(autotrain_url, timeout=0.2)
                    if response.status_code == 200:
                        server_ready = True
                        webbrowser.open(autotrain_url)
                        message = f"\nSuccessfully launched AutoTrain UI. It should now be open at {autotrain_url}."
                        print(message)
                        log_output += message
                        yield log_output, gr.update(interactive=False), gr.update(visible=True)
                except (requests.ConnectionError, requests.Timeout):
                    pass

            if not server_ready and time.time() - start_time > timeout:
                stop_autotrain_ui()
                message = f"\nAutoTrain UI failed to start within {timeout} seconds. The process has been stopped."
                print(message)
                log_output += message
                yield log_output, gr.update(visible=True), gr.update(visible=False)
                return

            time.sleep(0.1)

        reader_thread.join(timeout=1)
        try:
            while True:
                line = output_queue.get_nowait()
                print(line, end='')
                log_output += line
        except queue.Empty:
            pass
        
        message = f"\nAutoTrain UI process terminated with exit code {AUTOTRAIN_PROCESS.returncode}."
        print(message)
        log_output += message
        AUTOTRAIN_PROCESS = None
        yield log_output, gr.update(visible=True), gr.update(visible=False)

    except Exception as e:
        message = f"Failed to launch AutoTrain UI: {e}"
        print(message)
        yield message, gr.update(visible=True), gr.update(visible=False)

def stop_autotrain_ui():
    """Stops the AutoTrain UI process and its children."""
    global AUTOTRAIN_PROCESS
    process = AUTOTRAIN_PROCESS
    if not process or process.poll() is not None:
        message = "AutoTrain UI process is not running or was already stopped."
        print(message)
        AUTOTRAIN_PROCESS = None
        return message, gr.update(visible=True), gr.update(visible=False)

    try:
        if sys.platform == "win32":
            # On Windows, use taskkill to forcefully terminate the process tree.
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            message = "AutoTrain UI process has been stopped."
        else:
            # On Unix-like systems, send SIGTERM to the process group.
            pgid = os.getpgid(process.pid)
            os.killpg(pgid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
                message = "AutoTrain UI process has been stopped."
            except subprocess.TimeoutExpired:
                os.killpg(pgid, signal.SIGKILL)
                message = "AutoTrain UI process did not stop gracefully and was killed."
    except (ProcessLookupError, subprocess.CalledProcessError):
        # ProcessLookupError: process already died (Unix).
        # CalledProcessError: taskkill failed, likely because process died (Windows).
        message = "AutoTrain UI process was already stopped."
    except Exception as e:
        message = f"An unexpected error occurred while stopping AutoTrain UI: {e}"
        print(message)
        return message, gr.update(visible=False), gr.update(visible=True)

    print(message)
    AUTOTRAIN_PROCESS = None
    return message, gr.update(visible=True), gr.update(visible=False)

def show_model_charts(model_dir):
    """Finds trainer_state.json, returns metric plots, and the model_dir for sync."""
    if not model_dir:
        return (None,) * 11 + (gr.update(visible=False), None)

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
        return (None,) * 11 + (gr.update(visible=False), model_dir)

    try:
        figures = util_plot_training_metrics(json_path)
        return (
            figures.get('Loss'), figures.get('Accuracy'), figures.get('Learning Rate'),
            figures.get('Gradient Norm'), figures.get('F1 Scores'), figures.get('Precision'),
            figures.get('Recall'), figures.get('Epoch'), figures.get('Eval Runtime'),
            figures.get('Eval Samples/sec'), figures.get('Eval Steps/sec'),
            gr.update(visible=True),
            model_dir
        )
    except Exception as e:
        print(f"Error generating plots for {json_path}: {e}")
        return (None,) * 11 + (gr.update(visible=False), model_dir)


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


def split_dataset(source_dir, train_zip_path, val_zip_path, test_zip_path, train_manifest_path, val_manifest_path, test_manifest_path, split_type, train_ratio, val_ratio, test_ratio, resample_train_set):
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
            resampling_applied = False

            # --- Handle training set resampling or default file copying ---
            if set_name == 'train' and resample_train_set:
                print("Applying memory-efficient SMOTE and RandomUnderSampler to the training set...")
                resampling_applied = True
                IMG_DIM = (224, 224)

                class_file_counts = {name: len(files) for name, files in classes.items()}
                if not class_file_counts:
                    print("Warning: No classes found for resampling. Skipping.")
                    resampling_applied = False
                else:
                    # 1. Determine target size for all classes using the median count
                    counts = np.array(list(class_file_counts.values()))
                    target_size = int(np.median(counts))
                    print(f"Balancing all classes to the median size: {target_size} samples.")

                    final_class_counts = {}

                    # 2. Process each class to meet the target size
                    for class_name, files_to_process in classes.items():
                        class_dir = os.path.join(set_dir, class_name)
                        os.makedirs(class_dir, exist_ok=True)
                        
                        num_original_samples = len(files_to_process)
                        
                        if num_original_samples > target_size:
                            # Undersample
                            print(f"Undersampling class '{class_name}' from {num_original_samples} to {target_size} samples.")
                            files_to_copy = random.sample(files_to_process, target_size)
                            for f in files_to_copy:
                                shutil.copy2(f, class_dir)
                                manifest_files.append(f"{class_name}/{os.path.basename(f)}".replace(os.sep, '/'))
                            final_class_counts[class_name] = target_size
                        
                        elif num_original_samples < target_size:
                            # Oversample
                            num_to_synthesize = target_size - num_original_samples
                            final_class_counts[class_name] = target_size

                            # Copy original files
                            for f in files_to_process:
                                shutil.copy2(f, class_dir)
                                manifest_files.append(f"{class_name}/{os.path.basename(f)}".replace(os.sep, '/'))

                            # Apply SMOTE if possible
                            k_neighbors = min(5, num_original_samples - 1) if num_original_samples > 1 else 0
                            if k_neighbors > 0 and num_to_synthesize > 0:
                                print(f"Applying SMOTE to class '{class_name}' to generate {num_to_synthesize} new samples...")
                                X_class = []
                                for f in files_to_process:
                                    try:
                                        n = np.fromfile(f, np.uint8)
                                        img = cv2.imdecode(n, cv2.IMREAD_COLOR)
                                        if img is not None:
                                            X_class.append(cv2.resize(img, IMG_DIM))
                                    except Exception as e:
                                        print(f"Warning: Could not load image {f} for SMOTE. Error: {e}")
                                
                                if len(X_class) > k_neighbors:
                                    X_class_flat = np.array(X_class).reshape(len(X_class), -1).astype(np.float32)
                                    nn = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(X_class_flat)
                                    indices = nn.kneighbors(X_class_flat, return_distance=False)[:, 1:]
                                    
                                    for i in range(num_to_synthesize):
                                        sample_idx = random.randint(0, len(X_class_flat) - 1)
                                        neighbor_idx = random.choice(indices[sample_idx])
                                        diff = X_class_flat[neighbor_idx] - X_class_flat[sample_idx]
                                        synthetic_sample_flat = X_class_flat[sample_idx] + random.random() * diff
                                        synthetic_img = synthetic_sample_flat.reshape(IMG_DIM[0], IMG_DIM[1], 3).astype(np.uint8)
                                        filename = f"synthetic_{i:05d}.png"
                                        filepath = os.path.join(class_dir, filename)
                                        cv2.imwrite(filepath, synthetic_img)
                                        manifest_files.append(f"{class_name}/{filename}".replace(os.sep, '/'))
                                else:
                                    print(f"Warning: Not enough valid images loaded for '{class_name}' to perform SMOTE. Reverting to original size.")
                                    final_class_counts[class_name] = num_original_samples
                            else:
                                final_class_counts[class_name] = num_original_samples
                                print(f"Skipping SMOTE for class '{class_name}' (not enough samples).")
                        
                        else: # num_original_samples == target_size
                            # No change needed, just copy files
                            print(f"Class '{class_name}' is already at the target size of {target_size}. Copying files.")
                            for f in files_to_process:
                                shutil.copy2(f, class_dir)
                                manifest_files.append(f"{class_name}/{os.path.basename(f)}".replace(os.sep, '/'))
                            final_class_counts[class_name] = target_size

                    # 3. Update the main `included_classes` dict with new counts for the manifest
                    for class_name, count in final_class_counts.items():
                        if class_name in included_classes:
                            included_classes[class_name]['splits']['train'] = count
            else:
                # --- Default file copying for validation/test or if resampling is skipped ---
                for class_name, files_to_copy in classes.items():
                    class_dir = os.path.join(set_dir, class_name)
                    os.makedirs(class_dir, exist_ok=True)
                    for f in files_to_copy:
                        shutil.copy2(f, class_dir)
                        file_name = os.path.basename(f)
                        manifest_files.append(f"{class_name}/{file_name}".replace(os.sep, '/'))

            # --- Build manifest content with summary ---
            manifest_content = [f"# {set_name.capitalize()} Set Manifest"]
            if resampling_applied:
                manifest_content.append("\n*This training set has been resampled to balance class distribution. Oversampling (SMOTE) and undersampling have been applied to bring each class to the median size.*")

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


def get_model_choices():
    """
    Finds model directories, identifies the latest checkpoint in each, and returns
    a list of (display_name, path_to_checkpoint) tuples for the dropdown.
    """
    choices = []
    try:
        app_dir = os.path.dirname(os.path.abspath(__file__))
        model_parent_dirs = [d for d in os.listdir(app_dir) if os.path.isdir(os.path.join(app_dir, d)) and d.startswith('Model-')]

        for model_dir_name in model_parent_dirs:
            model_dir_path = os.path.join(app_dir, model_dir_name)
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
                display_name = model_dir_name
                choices.append((display_name, path))
            else:
                # If no checkpoints, maybe the model is in the root. Add it as a choice.
                # Check for a config file to be sure it's a model directory.
                if os.path.exists(os.path.join(model_dir_path, 'config.json')):
                    choices.append((model_dir_name, model_dir_path))

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
    
    return gr.update(choices=choices, value=new_value), gr.update(choices=choices, value=new_value), gr.update(choices=choices, value=new_value), gr.update(choices=choices, value=new_value)
