import os
import shutil
import re
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
