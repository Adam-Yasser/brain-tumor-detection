"""
Prepare the dataset for training.

This script:
1. Removes suspect (edge-detected) images from training set
2. Creates a train/validation split (80/20)
3. Organizes data into data/splits/ directory

Run once before training:
    python scripts/prepare_data.py
"""

import shutil
import random
from pathlib import Path
from PIL import Image
import numpy as np

# Reproducibility — same split every time
random.seed(42)

# Paths
RAW_DIR = Path("data/raw/archive")
SPLITS_DIR = Path("data/splits")
TRAIN_DIR = RAW_DIR / "Training"
TEST_DIR = RAW_DIR / "Testing"

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]


def find_suspect_images(directory):
    """Find edge-detected/processed images using pixel analysis."""
    suspect = []
    for img_path in directory.iterdir():
        try:
            img = Image.open(img_path).convert("L")
            pixels = np.array(img)
            bright_ratio = np.mean(pixels > 200)
            diff = np.abs(np.diff(pixels.astype(float), axis=1))
            mean_diff = np.mean(diff)
            if bright_ratio > 0.15 and mean_diff > 30:
                suspect.append(img_path)
        except Exception as e:
            print(f"  Error reading {img_path}: {e}")
            suspect.append(img_path)
    return suspect


def prepare_data(val_ratio=0.2):
    """Clean data and create train/val/test splits."""

    # Clean output directory
    if SPLITS_DIR.exists():
        shutil.rmtree(SPLITS_DIR)

    # Create split directories
    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            (SPLITS_DIR / split / cls).mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("PREPARING DATASET")
    print("=" * 50)

    # --- Step 1: Process Training Data → train + val ---
    print("\n--- Processing Training Set ---")
    total_removed = 0
    total_train = 0
    total_val = 0

    for cls in CLASSES:
        cls_dir = TRAIN_DIR / cls

        # Find and exclude suspect images
        suspect = find_suspect_images(cls_dir) if cls == "notumor" else []
        suspect_names = {p.name for p in suspect}

        # Get clean image paths
        clean_images = [
            p for p in cls_dir.iterdir()
            if p.name not in suspect_names
        ]

        # Shuffle and split
        random.shuffle(clean_images)
        val_count = int(len(clean_images) * val_ratio)
        val_images = clean_images[:val_count]
        train_images = clean_images[val_count:]

        # Copy files to split directories
        for img_path in train_images:
            shutil.copy2(img_path, SPLITS_DIR / "train" / cls / img_path.name)
        for img_path in val_images:
            shutil.copy2(img_path, SPLITS_DIR / "val" / cls / img_path.name)

        total_removed += len(suspect)
        total_train += len(train_images)
        total_val += len(val_images)

        print(f"  {cls:15s} → removed: {len(suspect):2d} | train: {len(train_images)} | val: {len(val_images)}")

    # --- Step 2: Copy Testing Data ---
    print("\n--- Processing Testing Set ---")
    total_test = 0

    for cls in CLASSES:
        cls_dir = TEST_DIR / cls
        for img_path in cls_dir.iterdir():
            shutil.copy2(img_path, SPLITS_DIR / "test" / cls / img_path.name)
            total_test += 1
        print(f"  {cls:15s} → test: {len(list(cls_dir.iterdir()))}")

    # --- Summary ---
    print(f"\n{'=' * 50}")
    print(f"SUMMARY")
    print(f"{'=' * 50}")
    print(f"  Suspect images removed: {total_removed}")
    print(f"  Training:   {total_train} images")
    print(f"  Validation: {total_val} images")
    print(f"  Testing:    {total_test} images")
    print(f"  Total:      {total_train + total_val + total_test} images")
    print(f"\n  Output: {SPLITS_DIR.resolve()}")


if __name__ == "__main__":
    prepare_data()