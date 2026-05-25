import os
import shutil
import random
import kagglehub
from pathlib import Path

def setup_lfw_subset(subset_size=3000, target_dir="data/lfw_clean"):
    print("Downloading LFW dataset from Kaggle...")
    # This automatically downloads and caches the dataset, returning the path
    dataset_path = kagglehub.dataset_download("jessicali9530/lfw-dataset")
    print(f"Dataset downloaded/cached at: {dataset_path}")
    
    # Handle possible nested directory structures from Kaggle unzipping
    deepfunneled_dir = os.path.join(dataset_path, "lfw-deepfunneled", "lfw-deepfunneled")
    if not os.path.exists(deepfunneled_dir):
        deepfunneled_dir = os.path.join(dataset_path, "lfw-deepfunneled")
        if not os.path.exists(deepfunneled_dir):
            deepfunneled_dir = dataset_path # Fallback to root

    all_images = []
    for root, _, files in os.walk(deepfunneled_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_images.append(os.path.join(root, file))
                
    print(f"Found {len(all_images)} total images.")
    
    if len(all_images) == 0:
        print("Error: No images found. Check the dataset path structure.")
        return
        
    # Shuffle and select a subset
    random.seed(42)
    random.shuffle(all_images)
    subset = all_images[:min(subset_size, len(all_images))]
    
    # Create the target directory and copy files while preserving identity folders
    if os.path.exists(target_dir):
        print(f"Cleaning existing {target_dir}...")
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)
    
    print(f"Copying {len(subset)} images to {target_dir}...")
    for src_path in subset:
        person_name = os.path.basename(os.path.dirname(src_path))
        dest_person_dir = os.path.join(target_dir, person_name)
        os.makedirs(dest_person_dir, exist_ok=True)
        
        dest_path = os.path.join(dest_person_dir, os.path.basename(src_path))
        shutil.copy2(src_path, dest_path)
        
    print(f"\n[Success] Prepared {len(subset)} clean in-the-wild images!")
    print("You can now run Phase 1: Generate adversarial attacks")

if __name__ == "__main__":
    setup_lfw_subset()
