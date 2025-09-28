#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ExoGlove Training Script - Stable YOLOv8 version
"""

import torch
from ultralytics import YOLO

def train_model():
    print("Starting ExoGlove training with stable YOLOv8...")
    
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
    
    # Train with stable parameters
    results = model.train(
        data='data.yaml',
        epochs=50,
        imgsz=640,
        batch=8,
        device=device,
        project='runs/train',
        name='exoglove_stable',
        save=True,
        patience=15,
        verbose=True,
        plots=True,
        val=True
    )
    
    print("Training completed!")
    return results

if __name__ == "__main__":
    try:
        results = train_model()
        print("Training successful!")
        
        # Export for Pi deployment
        print("\nExporting model for Pi deployment...")
        model = YOLO('runs/train/exoglove_stable/weights/best.pt')
        
        # Export to different formats
        try:
            onnx_path = model.export(format='onnx', imgsz=640)
            print(f"ONNX export: {onnx_path}")
        except Exception as e:
            print(f"ONNX export failed: {e}")
        
        try:
            torchscript_path = model.export(format='torchscript', imgsz=640)
            print(f"TorchScript export: {torchscript_path}")
        except Exception as e:
            print(f"TorchScript export failed: {e}")
        
        print("\nTraining and export completed successfully!")
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
