import os
import shutil
import re
import random
import gradio as gr

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

def split_paired_dataset_custom(source_dir, output_dir, val_ratio, min_items=5):
    if not source_dir or not os.path.isdir(source_dir):
        raise gr.Error("Please provide a valid source directory.")
    if not output_dir:
        raise gr.Error("Please provide an output directory.")
    
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    stats = {
        "species_count": 0,
        "train_herbarium": 0,
        "train_photo": 0,
        "val_photo": 0,
        "padded_files": 0
    }
    
    def safe_copy(file_list, dest_dir):
        count = 0
        for src_path in file_list:
            filename = os.path.basename(src_path)
            dest_path = os.path.join(dest_dir, filename)
            
            # If exists (due to duplication), append suffix
            counter = 1
            base, ext = os.path.splitext(filename)
            while os.path.exists(dest_path):
                dest_path = os.path.join(dest_dir, f"{base}_copy{counter}{ext}")
                counter += 1
            
            shutil.copy2(src_path, dest_path)
            count += 1
        return count

    # Iterate over species folders
    for species_name in os.listdir(source_dir):
        species_path = os.path.join(source_dir, species_name)
        if not os.path.isdir(species_path):
            continue
            
        herbarium_files = []
        photo_files = []
        
        for filename in os.listdir(species_path):
            fname_lower = filename.lower()
            if not fname_lower.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')):
                continue
                
            file_path = os.path.join(species_path, filename)
            
            if "herbarium" in fname_lower:
                herbarium_files.append(file_path)
            elif "photo" in fname_lower:
                photo_files.append(file_path)
        
        # --- Logic to ensure min items per set (with padding) ---
        
        # 1. Validation Set (Photos Only)
        # Calculate target size
        req_val_photos = int(len(photo_files) * (val_ratio / 100.0))
        if req_val_photos < min_items: req_val_photos = min_items
        
        # Pad photos if insufficient
        while len(photo_files) < req_val_photos:
            if not photo_files: break # Cannot pad if no photos exist
            photo_files.append(random.choice(photo_files))
            stats["padded_files"] += 1
            
        # Split Photos
        random.shuffle(photo_files)
        val_photos = photo_files[:req_val_photos]
        train_photos = photo_files[req_val_photos:]
        
        # 2. Training Set (Herbarium + Remaining Photos)
        # Check if we have enough items for training
        current_train_count = len(herbarium_files) + len(train_photos)
        
        while current_train_count < min_items:
            # Pad Training set
            if herbarium_files:
                herbarium_files.append(random.choice(herbarium_files))
                stats["padded_files"] += 1
                current_train_count += 1
            elif train_photos:
                train_photos.append(random.choice(train_photos))
                stats["padded_files"] += 1
                current_train_count += 1
            elif val_photos:
                # Fallback: Duplicate a validation photo into training
                train_photos.append(random.choice(val_photos))
                stats["padded_files"] += 1
                current_train_count += 1
            else:
                break # No files to duplicate
        
        # --- Execution ---
        stats["species_count"] += 1
        
        train_species_dir = os.path.join(train_dir, species_name)
        val_species_dir = os.path.join(val_dir, species_name)
        os.makedirs(train_species_dir, exist_ok=True)
        os.makedirs(val_species_dir, exist_ok=True)
        
        stats["train_herbarium"] += safe_copy(herbarium_files, train_species_dir)
        stats["train_photo"] += safe_copy(train_photos, train_species_dir)
        stats["val_photo"] += safe_copy(val_photos, val_species_dir)

    result_msg = (f"Processing complete.\n"
                  f"Species processed: {stats['species_count']}\n"
                  f"Files padded (duplicated): {stats['padded_files']}\n"
                  f"Train images: {stats['train_herbarium']} (Herbarium) + {stats['train_photo']} (Photo)\n"
                  f"Val images: {stats['val_photo']} (Photo only)\n"
                  f"Output at: {output_dir}")
        
    return result_msg

def split_hybrid_dataset(source_dir, output_dir, val_ratio, min_items=5):
    if not source_dir or not os.path.isdir(source_dir):
        raise gr.Error("Please provide a valid source directory.")
    if not output_dir:
        raise gr.Error("Please provide an output directory.")
    
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    stats = {
        "species_count": 0,
        "train_count": 0,
        "val_count": 0,
        "padded_files": 0
    }
    
    def safe_copy(file_list, dest_dir):
        count = 0
        for src_path in file_list:
            filename = os.path.basename(src_path)
            dest_path = os.path.join(dest_dir, filename)
            
            # If exists (due to duplication), append suffix
            counter = 1
            base, ext = os.path.splitext(filename)
            while os.path.exists(dest_path):
                dest_path = os.path.join(dest_dir, f"{base}_copy{counter}{ext}")
                counter += 1
            
            shutil.copy2(src_path, dest_path)
            count += 1
        return count

    # Iterate over species folders
    for species_name in os.listdir(source_dir):
        species_path = os.path.join(source_dir, species_name)
        if not os.path.isdir(species_path):
            continue
            
        herbarium_files = []
        photo_files = []
        
        for filename in os.listdir(species_path):
            fname_lower = filename.lower()
            if not fname_lower.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')):
                continue
                
            file_path = os.path.join(species_path, filename)
            
            if "herbarium" in fname_lower:
                herbarium_files.append(file_path)
            elif "photo" in fname_lower:
                photo_files.append(file_path)
        
        # --- Logic for Hybrid Split ---
        total_files = len(photo_files) + len(herbarium_files)
        if total_files == 0: continue

        # 1. Calculate Validation Target
        req_val = int(total_files * (val_ratio / 100.0))
        if req_val < min_items: req_val = min_items
        
        val_files = []
        train_files = []
        
        # 2. Fill Validation (Prioritise Photos)
        random.shuffle(photo_files)
        random.shuffle(herbarium_files)
        
        # Take as many photos as possible up to req_val
        while len(val_files) < req_val and photo_files:
            val_files.append(photo_files.pop(0))
            
        # If still need more, take herbarium
        while len(val_files) < req_val and herbarium_files:
            val_files.append(herbarium_files.pop(0))
            
        # If STILL need more (not enough unique files total), pad with duplicates
        # Priority for padding Val: Photos -> Herbarium
        while len(val_files) < req_val:
            # Create pool of available files to clone from
            pool = val_files + photo_files + herbarium_files
            if not pool: break # Should not happen if total_files > 0
            
            # Prefer photos if any exist in the pool
            photo_pool = [f for f in pool if "photo" in os.path.basename(f).lower()]
            if photo_pool:
                val_files.append(random.choice(photo_pool))
            else:
                val_files.append(random.choice(pool))
            stats["padded_files"] += 1

        # 3. Fill Training (Remaining files)
        train_files.extend(photo_files)
        train_files.extend(herbarium_files)
        
        # 4. Check Training Minimum
        while len(train_files) < min_items:
            # Pad Training set
            # Priority: Herbarium -> Train Photos -> Val Photos
            
            # Check what we have in train_files first
            herb_in_train = [f for f in train_files if "herbarium" in os.path.basename(f).lower()]
            photo_in_train = [f for f in train_files if "photo" in os.path.basename(f).lower()]
            
            if herb_in_train:
                train_files.append(random.choice(herb_in_train))
            elif photo_in_train:
                train_files.append(random.choice(photo_in_train))
            elif val_files:
                train_files.append(random.choice(val_files))
            else:
                break
            stats["padded_files"] += 1

        # --- Execution ---
        stats["species_count"] += 1
        
        train_species_dir = os.path.join(train_dir, species_name)
        val_species_dir = os.path.join(val_dir, species_name)
        os.makedirs(train_species_dir, exist_ok=True)
        os.makedirs(val_species_dir, exist_ok=True)
        
        stats["train_count"] += safe_copy(train_files, train_species_dir)
        stats["val_count"] += safe_copy(val_files, val_species_dir)

    result_msg = (f"Processing complete.\n"
                  f"Species processed: {stats['species_count']}\n"
                  f"Files padded (duplicated): {stats['padded_files']}\n"
                  f"Train images: {stats['train_count']}\n"
                  f"Val images: {stats['val_count']}\n"
                  f"Output at: {output_dir}")
        
    return result_msg

def separate_paired_species(source_dir, output_dir):
    if not source_dir or not os.path.isdir(source_dir):
        raise gr.Error("Please provide a valid source directory.")
    if not output_dir:
        raise gr.Error("Please provide an output directory.")

    paired_dir = os.path.join(output_dir, "paired")
    unpaired_dir = os.path.join(output_dir, "unpaired")
    
    os.makedirs(paired_dir, exist_ok=True)
    os.makedirs(unpaired_dir, exist_ok=True)
    
    paired_count = 0
    unpaired_count = 0
    
    # Iterate over species folders
    for species_name in os.listdir(source_dir):
        species_path = os.path.join(source_dir, species_name)
        if not os.path.isdir(species_path):
            continue
            
        has_herbarium = False
        has_photo = False
        
        # Check files in the species folder
        for filename in os.listdir(species_path):
            fname_lower = filename.lower()
            if not fname_lower.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')):
                continue
                
            if "herbarium" in fname_lower:
                has_herbarium = True
            if "photo" in fname_lower:
                has_photo = True
                
            if has_herbarium and has_photo:
                break
        
        if has_herbarium and has_photo:
            dest_path = os.path.join(paired_dir, species_name)
            paired_count += 1
        else:
            dest_path = os.path.join(unpaired_dir, species_name)
            unpaired_count += 1
            
        # Copy directory
        try:
            shutil.copytree(species_path, dest_path, dirs_exist_ok=True)
        except Exception as e:
            print(f"Error copying {species_name}: {e}")
        
    return f"Processing complete.\nPaired species: {paired_count}\nUnpaired species: {unpaired_count}\nOutput at: {output_dir}"

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
