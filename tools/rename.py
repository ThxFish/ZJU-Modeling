import os
import glob
from pathlib import Path

def rename_dataset(dataset_dir="dataset"):
    dataset_path = Path(dataset_dir)
    splits = ["train", "valid", "test"]
    
    global_idx = 1
    
    for split in splits:
        images_dir = dataset_path / split / "images"
        labels_dir = dataset_path / split / "labels"
        
        if not images_dir.exists() or not labels_dir.exists():
            print(f"Skipping {split} as images or labels directory is missing.")
            continue
            
        print(f"Processing {split} set...")
        
        # Get all image files
        image_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
            image_files.extend(images_dir.glob(ext))
            
        image_files = sorted(image_files)
        
        for img_path in image_files:
            # Check if corresponding label exists
            label_name = img_path.stem + ".txt"
            label_path = labels_dir / label_name
            
            if not label_path.exists():
                print(f"Warning: Label not found for image {img_path.name}")
                continue
                
            # New names
            new_img_name = f"{global_idx}{img_path.suffix}"
            new_label_name = f"{global_idx}.txt"
            
            new_img_path = images_dir / new_img_name
            new_label_path = labels_dir / new_label_name
            
            # Handle potential conflicts if a file already has the target name
            # A safe way is to first rename to temporary names, then to target names,
            # but since we are iterating sorted existing names, we can just rename if it's different.
            # To be absolutely safe against overwriting during rename, we rename all to temp first.
            
            temp_img_path = images_dir / f"temp_{global_idx}{img_path.suffix}"
            temp_label_path = labels_dir / f"temp_{global_idx}.txt"
            
            os.rename(img_path, temp_img_path)
            os.rename(label_path, temp_label_path)
            
            global_idx += 1

    # Second pass: remove 'temp_' prefix
    print("Finalizing renaming...")
    global_idx = 1
    for split in splits:
        images_dir = dataset_path / split / "images"
        labels_dir = dataset_path / split / "labels"
        
        if not images_dir.exists() or not labels_dir.exists():
            continue
            
        temp_images = list(images_dir.glob("temp_*"))
        for temp_img in temp_images:
            final_img = temp_img.parent / temp_img.name.replace("temp_", "")
            os.rename(temp_img, final_img)
            
        temp_labels = list(labels_dir.glob("temp_*"))
        for temp_label in temp_labels:
            final_label = temp_label.parent / temp_label.name.replace("temp_", "")
            os.rename(temp_label, final_label)
            
    print(f"Successfully renamed files up to index {global_idx - 1}.")

if __name__ == "__main__":
    rename_dataset()
