import os

def main():
    # Paths relative to the repository root
    test_dir = os.path.join("Dataset-PlantCLEF-2020-Challenge", "Test")
    groundtruth_path = os.path.join("AML-dataset", "AML_project_herbarium_dataset", "list", "groundtruth.txt")
    species_list_path = os.path.join("AML-dataset", "AML_project_herbarium_dataset", "list", "species_list.txt")

    if not os.path.exists(test_dir):
        print(f"Test directory not found: {test_dir}")
        return

    if not os.path.exists(groundtruth_path):
        print(f"Groundtruth file not found: {groundtruth_path}")
        return

    if not os.path.exists(species_list_path):
        print(f"Species list file not found: {species_list_path}")
        return

    # Load species list
    print(f"Reading species list from {species_list_path}...")
    class_to_species = {}
    with open(species_list_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split(';')
            if len(parts) >= 2:
                class_id = parts[0].strip()
                species_name = parts[1].strip()
                class_to_species[class_id] = species_name

    # Load groundtruth
    print(f"Reading groundtruth from {groundtruth_path}...")
    id_to_class = {}
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

    print(f"Found {len(id_to_class)} entries in groundtruth.")
    
    renamed_count = 0
    
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
            # print(f"Skipping {filename}: ID {img_id} not found in groundtruth.")
            continue
            
        class_id = id_to_class[img_id]
        species_name = class_to_species.get(class_id, "Unknown")
        
        # Sanitize species name for filename
        safe_species_name = "".join([c if c.isalnum() else "_" for c in species_name])
        
        # Check if already renamed (starts with class_id)
        if filename.startswith(f"{class_id}_"):
             continue

        # Construct new filename
        # Format: {ClassID}_{SpeciesName}_{OriginalName}
        new_filename = f"{class_id}_{safe_species_name}_{filename}"
        
        source_path = os.path.join(test_dir, filename)
        dest_path = os.path.join(test_dir, new_filename)
        
        try:
            os.rename(source_path, dest_path)
            renamed_count += 1
        except OSError as e:
            print(f"Error renaming {filename}: {e}")

    print(f"Finished.")
    print(f"Renamed {renamed_count} files.")

if __name__ == "__main__":
    main()
