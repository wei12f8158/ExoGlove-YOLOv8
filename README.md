# ExoGlove YOLOv8 Object Detection

A real-time object detection system for ExoGlove using YOLOv8, optimized for Raspberry Pi 5 + IMX500 deployment.

## 🎯 Project Overview

This project detects 9 object classes relevant to ExoGlove applications:
- apple, ball, bottle, clip, glove, lid, plate, spoon, tape spool

## 📊 Dataset

- **Source**: Roboflow ExoGlove dataset
- **Images**: 11,701 total (10,215 train, 993 validation, 493 test)
- **Classes**: 9 object types
- **Format**: YOLO format annotations

## 🚀 Quick Start

### Training on MacBook

```bash
# Setup environment
python3 -m venv exoglove_env
source exoglove_env/bin/activate
pip install -r requirements.txt

# Train model
python train_from_scratch.py

# Monitor training
python monitor_training.py
```

### Deployment to Raspberry Pi 5

```bash
# Create deployment package
python pi_deployment_package.py

# Transfer to Pi
scp exoglove_pi_deployment.zip pi@<PI_IP>:/home/pi/

# On Pi: Install and run
ssh pi@<PI_IP>
cd /home/pi && unzip exoglove_pi_deployment.zip
pip install -r requirements_pi.txt
python exoglove_pi_inference.py
```

## 📁 Project Structure

```
ExoGlove/
├── data.yaml                 # Dataset configuration
├── train_from_scratch.py     # Training script
├── exoglove_pi_inference.py  # Pi inference script
├── monitor_training.py       # Training monitor
├── pi_deployment_package.py  # Deployment creator
├── requirements.txt          # MacBook dependencies
├── requirements_pi.txt       # Pi dependencies
├── DEPLOYMENT_GUIDE.md       # Pi setup guide
├── train/                    # Training images (ignored by git)
├── valid/                    # Validation images (ignored by git)
├── test/                     # Test images (ignored by git)
└── runs/                     # Training outputs (ignored by git)
```

## 🍓 Pi 5 + IMX500 Setup

See `DEPLOYMENT_GUIDE.md` for complete setup instructions.

### Hardware Requirements
- Raspberry Pi 5
- IMX500 camera module
- MicroSD card (32GB+)
- Power supply (5V/3A)

### Software Requirements
- Python 3.9+
- OpenCV
- ONNX Runtime
- Ultralytics YOLOv8

## 📈 Model Performance

- **Architecture**: YOLOv8n (nano)
- **Parameters**: 3M
- **Model Size**: ~6MB (ONNX)
- **Input**: 640x640 RGB images
- **FPS**: 15-20 on Pi 5
- **Classes**: 9 ExoGlove objects

## 🔧 Development

### Training Configuration
- **Epochs**: 30
- **Batch Size**: 8
- **Device**: Apple Silicon GPU (MPS)
- **Optimizer**: AdamW
- **Learning Rate**: Auto (0.000769)

### Model Export
```bash
# Export to ONNX for Pi deployment
python -c "
from ultralytics import YOLO
model = YOLO('runs/train/exoglove_no_val/weights/best.pt')
model.export(format='onnx', imgsz=640)
"
```

## 📝 License

This project uses the Roboflow ExoGlove dataset under CC BY 4.0 license.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📞 Support

For issues and questions, please open a GitHub issue.
