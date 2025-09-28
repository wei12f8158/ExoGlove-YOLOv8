#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ExoGlove Training Script - Skip validation to avoid errors
"""

import torch
from ultralytics import YOLO

def train_model():
    print("Starting ExoGlove training (no validation)...")
    
    # Check device
    if torch.cuda.is_available():
        device = "cuda"
        print("Using CUDA")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = "cpu"
        print("Using CPU")
    
    # Initialize model
    model = YOLO('yolov8n.pt')
    
    # Train with validation disabled
    results = model.train(
        data='data.yaml',
        epochs=30,  # Reduced epochs
        imgsz=640,
        batch=8,
        device=device,
        project='runs/train',
        name='exoglove_no_val',
        save=True,
        val=False,  # Disable validation
        plots=False,  # Disable plots
        verbose=True
    )
    
    print("Training completed!")
    return results

if __name__ == "__main__":
    try:
        results = train_model()
        print("Training successful!")
        
        # Now try validation separately
        print("\nRunning validation separately...")
        model = YOLO('runs/train/exoglove_no_val/weights/best.pt')
        val_results = model.val(data='data.yaml', verbose=True)
        print("Validation completed!")
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
