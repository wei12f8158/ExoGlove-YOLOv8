#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple script to fix ExoGlove dataset issues
"""

import os
import glob
from pathlib import Path

def fix_dataset():
    print("Fixing ExoGlove dataset...")
    
    # Check for empty label files
    empty_files = []
    for split in ['train', 'valid', 'test']:
        label_dir = Path(split) / "labels"
        if label_dir.exists():
            for label_file in label_dir.glob("*.txt"):
                if label_file.stat().st_size == 0:
                    empty_files.append(label_file)
    
    print(f"Found {len(empty_files)} empty label files")
    
    # Remove empty label files and corresponding images
    removed_count = 0
    for label_file in empty_files:
        # Find corresponding image file
        image_file = label_file.parent.parent / "images" / (label_file.stem + ".jpg")
        
        if image_file.exists():
            # Remove both files
            label_file.unlink()
            image_file.unlink()
            removed_count += 1
            print(f"Removed: {label_file.name} and {image_file.name}")
    
    print(f"Removed {removed_count} empty label/image pairs")
    
    # Count remaining files
    print("\nFinal dataset counts:")
    for split in ['train', 'valid', 'test']:
        image_dir = Path(split) / "images"
        label_dir = Path(split) / "labels"
        
        if image_dir.exists() and label_dir.exists():
            image_count = len(list(image_dir.glob("*.jpg")))
            label_count = len(list(label_dir.glob("*.txt")))
            print(f"  {split}: {image_count} images, {label_count} labels")
    
    return True

if __name__ == "__main__":
    success = fix_dataset()
    if success:
        print("\nDataset fixed successfully!")
        print("You can now run training again.")
    else:
        print("\nDataset still has issues.")
