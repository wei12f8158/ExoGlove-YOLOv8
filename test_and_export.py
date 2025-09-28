#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test and Export ExoGlove Models for Pi Deployment
"""

import torch
from ultralytics import YOLO
import os
from pathlib import Path

def test_model(model_path):
    """Test a trained model"""
    print(f"Testing model: {model_path}")
    
    try:
        model = YOLO(model_path)
        
        # Test on a sample image
        test_image = "train/images/1_20250221-133724-454_1_ball_1_jpg.rf.500bb720ed1d7437f9d0abda61757012.jpg"
        if os.path.exists(test_image):
            results = model(test_image, verbose=False)
            print(f"✅ Model inference successful!")
            print(f"Found {len(results[0].boxes) if results[0].boxes is not None else 0} detections")
            return True
        else:
            print("Test image not found, but model loaded successfully")
            return True
            
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def export_model(model_path, model_name):
    """Export model for Pi deployment"""
    print(f"\nExporting model: {model_path}")
    
    try:
        model = YOLO(model_path)
        
        # Export to different formats
        exports = []
        
        try:
            onnx_path = model.export(format='onnx', imgsz=640, optimize=True)
            exports.append(('ONNX', onnx_path))
            print(f"✅ ONNX export: {onnx_path}")
        except Exception as e:
            print(f"❌ ONNX export failed: {e}")
        
        try:
            torchscript_path = model.export(format='torchscript', imgsz=640)
            exports.append(('TorchScript', torchscript_path))
            print(f"✅ TorchScript export: {torchscript_path}")
        except Exception as e:
            print(f"❌ TorchScript export failed: {e}")
        
        return exports
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        return []

def main():
    """Main function"""
    print("🚀 Testing and Exporting ExoGlove Models")
    
    # Find available models
    model_paths = []
    for root, dirs, files in os.walk("runs/train/"):
        for file in files:
            if file in ["best.pt", "last.pt"]:
                model_paths.append(os.path.join(root, file))
    
    if not model_paths:
        print("❌ No trained models found!")
        return
    
    print(f"Found {len(model_paths)} trained models:")
    for path in model_paths:
        print(f"  - {path}")
    
    # Test and export each model
    successful_models = []
    
    for model_path in model_paths:
        model_name = Path(model_path).parent.parent.name
        
        print(f"\n{'='*60}")
        print(f"Processing: {model_name}")
        print(f"{'='*60}")
        
        # Test model
        if test_model(model_path):
            # Export model
            exports = export_model(model_path, model_name)
            
            if exports:
                successful_models.append({
                    'name': model_name,
                    'path': model_path,
                    'exports': exports
                })
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 SUMMARY")
    print(f"{'='*60}")
    
    if successful_models:
        print(f"✅ Successfully processed {len(successful_models)} models:")
        
        for model in successful_models:
            print(f"\n🎯 {model['name']}:")
            print(f"   Model: {model['path']}")
            for format_name, export_path in model['exports']:
                print(f"   {format_name}: {export_path}")
        
        print(f"\n🍓 Ready for Pi Deployment!")
        print(f"Copy these files to your Raspberry Pi 5:")
        print(f"  - exoglove_pi_inference.py")
        print(f"  - requirements_pi.txt") 
        print(f"  - DEPLOYMENT_GUIDE.md")
        print(f"  - One of the trained models above")
        
    else:
        print("❌ No models could be processed successfully")

if __name__ == "__main__":
    main()
