#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fix ExoGlove Dataset Issues
- Remove empty label files
- Validate label format
- Fix class ID issues
"""

import os
import glob
from pathlib import Path

def fix_dataset():
    """Fix dataset issues"""
    print("Fixing ExoGlove dataset...")
    
    # Check for empty label files
    empty_files = []
    for split in ['train', 'valid', 'test']:
        label_dir = Path(f"{split}/labels")
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
    
    # Validate remaining label files
    print("Validating remaining label files...")
    
    total_files = 0
    valid_files = 0
    invalid_files = []
    
    for split in ['train', 'valid', 'test']:
        label_dir = Path(f"{split}/labels")
        if label_dir.exists():
            for label_file in label_dir.glob("*.txt"):
                total_files += 1
                
                try:
                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                    
                    # Check if file has valid content
                    if not lines:
                        invalid_files.append(label_file)
                        continue
                    
                    # Validate each line
                    for line_num, line in enumerate(lines, 1):
                        line = line.strip()
                        if not line:
                            continue
                        
                        parts = line.split()
                        if len(parts) != 5:
                            invalid_files.append(f"{label_file}:{line_num}")
                            continue
                        
                        try:
                            class_id = int(parts[0])
                            if class_id < 0 or class_id > 8:
                                invalid_files.append(f"{label_file}:{line_num} - invalid class ID: {class_id}")
                                continue
                            
                            # Check coordinates
                            for coord in parts[1:]:
                                coord_val = float(coord)
                                if coord_val < 0 or coord_val > 1:
                                    invalid_files.append(f"{label_file}:{line_num} - invalid coordinate: {coord_val}")
                                    break
                        except ValueError:
                            invalid_files.append(f"{label_file}:{line_num} - invalid format")
                            continue
                    
                    valid_files += 1
                    
                except Exception as e:
                    invalid_files.append(f"{label_file} - error: {e}")
    
    print(f"Validation Results:")
    print(f"  Total files: {total_files}")
    print(f"  Valid files: {valid_files}")
    print(f"  Invalid files: {len(invalid_files)}")
    
    if invalid_files:
        print("Invalid files found:")
        for invalid in invalid_files[:10]:  # Show first 10
            print(f"  - {invalid}")
        if len(invalid_files) > 10:
            print(f"  ... and {len(invalid_files) - 10} more")
    
    # Count remaining files
    print("\nFinal dataset counts:")
    for split in ['train', 'valid', 'test']:
        image_dir = Path(f"{split}/images")
        label_dir = Path(f"{split}/labels")
        
        if image_dir.exists() and label_dir.exists():
            image_count = len(list(image_dir.glob("*.jpg")))
            label_count = len(list(label_dir.glob("*.txt")))
            print(f"  {split}: {image_count} images, {label_count} labels")
    
    return len(invalid_files) == 0

if __name__ == "__main__":
    success = fix_dataset()
    if success:
        print("\nDataset fixed successfully!")
        print("You can now run training again.")
    else:
        print("\nDataset still has issues. Please check the invalid files.")
