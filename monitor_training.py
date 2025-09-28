#!/usr/bin/env python3
"""
Monitor training progress for ExoGlove YOLOv8 model
"""

import os
import time
import glob
from pathlib import Path

def monitor_training():
    """Monitor training progress"""
    print("🔍 Monitoring ExoGlove Training Progress...")
    print("=" * 50)
    
    # Find the latest training run
    train_dirs = glob.glob("runs/train/exoglove_from_scratch*")
    if not train_dirs:
        print("❌ No training runs found!")
        return
    
    latest_run = max(train_dirs, key=os.path.getmtime)
    print(f"📁 Latest run: {latest_run}")
    
    # Check for results.csv
    results_file = os.path.join(latest_run, "results.csv")
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            lines = f.readlines()
            if len(lines) > 1:
                print(f"✅ Training progress: {len(lines)-1} epochs completed")
                print("📊 Latest epoch results:")
                print(lines[-1].strip())
            else:
                print("⏳ Training started but no epochs completed yet")
    else:
        print("⏳ Training initializing...")
    
    # Check for weights
    weights_dir = os.path.join(latest_run, "weights")
    if os.path.exists(weights_dir):
        weight_files = glob.glob(os.path.join(weights_dir, "*.pt"))
        if weight_files:
            print(f"💾 Model weights found: {len(weight_files)} files")
            for wf in weight_files:
                size_mb = os.path.getsize(wf) / (1024*1024)
                print(f"   - {os.path.basename(wf)}: {size_mb:.1f} MB")
        else:
            print("⏳ No weights saved yet")
    
    # Check training log
    if os.path.exists("training.log"):
        print("\n📝 Latest training log entries:")
        with open("training.log", 'r') as f:
            lines = f.readlines()
            # Show last 5 lines
            for line in lines[-5:]:
                print(f"   {line.strip()}")

if __name__ == "__main__":
    monitor_training()
