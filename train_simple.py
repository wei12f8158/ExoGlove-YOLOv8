#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple ExoGlove Training Script
"""

import torch
from ultralytics import YOLO

def train_model():
    print("Starting ExoGlove training...")
    
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
    
    # Train with simple parameters
    results = model.train(
        data='data.yaml',
        epochs=50,  # Reduced epochs for faster training
        imgsz=640,
        batch=8,    # Reduced batch size
        device=device,
        project='runs/train',
        name='exoglove_simple',
        save=True,
        patience=10,  # Early stopping
        verbose=True
    )
    
    print("Training completed!")
    return results

if __name__ == "__main__":
    try:
        results = train_model()
        print("Training successful!")
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
