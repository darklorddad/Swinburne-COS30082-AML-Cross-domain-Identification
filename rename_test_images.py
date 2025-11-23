import os

def main():
    # Paths relative to the repository root
    manifest_path = os.path.join("Dataset-PlantCLEF-2020-Challenge", "Test-manifest.md")
    test_dir = os.path.join("Dataset-PlantCLEF-2020-Challenge", "Test")

    if not os.path.exists(manifest_path):
        print(f"Manifest file not found: {manifest_path}")
        return

    if not os.path.exists(test_dir):
        print(f"Test directory not found: {test_dir}")
        return

    print(f"Reading manifest from {manifest_path}...")
    with open(manifest_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    print(f"Found {len(lines)} entries in manifest.")
    
    renamed_count = 0
    missing_count = 0

    for filename in lines:
        # Expected format in manifest: test_unknown_no_pair_{ID}.jpg
        # We assume the current file on disk is named {ID}.jpg
        
        if not filename.endswith(".jpg"):
            continue
            
        # Extract ID from the filename
        # Example: test_unknown_no_pair_1000.jpg -> 1000.jpg
        parts = filename.split('_')
        if len(parts) < 2:
            print(f"Skipping entry with unexpected format: {filename}")
            continue
            
        # The last part should be the ID with extension
        id_filename = parts[-1] 
        
        source_path = os.path.join(test_dir, id_filename)
        dest_path = os.path.join(test_dir, filename)
        
        if os.path.exists(source_path):
            try:
                os.rename(source_path, dest_path)
                renamed_count += 1
            except OSError as e:
                print(f"Error renaming {id_filename}: {e}")
        elif os.path.exists(dest_path):
            # File already has the correct name
            pass
        else:
            # Neither source nor destination exists
            missing_count += 1

    print(f"Finished.")
    print(f"Renamed {renamed_count} files.")
    if missing_count > 0:
        print(f"Could not find {missing_count} files (they might be missing or named differently).")

if __name__ == "__main__":
    main()
