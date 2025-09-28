# ExoGlove Deployment Guide for Raspberry Pi 5 + IMX500

## Prerequisites
- Raspberry Pi 5 with 8GB RAM (recommended)
- IMX500 NPU module
- Camera module (USB or Pi Camera)
- MicroSD card (32GB+ recommended)
- Raspberry Pi OS (64-bit)

## Installation Steps

### 1. Update Pi OS
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip python3-venv git -y
```

### 2. Install Python Dependencies
```bash
# Create virtual environment
python3 -m venv exoglove_env
source exoglove_env/bin/activate

# Install requirements
pip install -r requirements_pi.txt
```

### 3. Install IMX500 Drivers (if using NPU)
```bash
# Follow IMX500 installation guide
# This may require specific drivers from manufacturer
```

### 4. Test Camera
```bash
# Test USB camera
lsusb | grep -i camera

# Test Pi camera (if using)
vcgencmd get_camera
```

### 5. Run Inference
```bash
# Copy your trained model to Pi
scp best.pt pi@your-pi-ip:/home/pi/exoglove/

# Run inference
python3 exoglove_pi_inference.py
```

## Performance Optimization

### For Better Performance:
1. **Use SSD**: Boot from USB SSD instead of SD card
2. **Increase GPU memory**: `sudo raspi-config` > Advanced > Memory Split > 128
3. **Disable unnecessary services**: 
   ```bash
   sudo systemctl disable bluetooth
   sudo systemctl disable hciuart
   ```
4. **Use ONNX export**: For faster inference with ONNX Runtime

### Expected Performance:
- **CPU inference**: ~200-500ms per frame
- **GPU inference**: ~50-150ms per frame  
- **NPU inference**: ~10-50ms per frame (with IMX500)

## Troubleshooting

### Common Issues:
1. **Camera not detected**: Check USB connection, try different camera
2. **Low FPS**: Reduce input resolution, use smaller model
3. **Memory issues**: Use YOLOv8n instead of larger models
4. **Import errors**: Ensure all dependencies are installed

### Debug Commands:
```bash
# Check system resources
htop
free -h
df -h

# Check camera
v4l2-ctl --list-devices

# Test inference speed
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
```
