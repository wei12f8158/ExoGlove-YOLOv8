#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ExoGlove Real-time Inference Script for Raspberry Pi 5 + IMX500
Optimized for edge deployment
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
import json
from pathlib import Path

class ExoGloveInference:
    def __init__(self, model_path, confidence_threshold=0.5):
        """Initialize inference engine"""
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.class_names = ['apple', 'ball', 'bottle', 'clip', 'glove', 'lid', 'plate', 'spoon', 'tape spool']
        
    def preprocess_image(self, image):
        """Preprocess image for inference"""
        # Resize to model input size
        resized = cv2.resize(image, (640, 640))
        return resized
    
    def postprocess_results(self, results):
        """Process YOLO results into readable format"""
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i, box in enumerate(boxes):
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    if confidence >= self.confidence_threshold:
                        detection = {
                            'class': self.class_names[class_id],
                            'confidence': float(confidence),
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'class_id': class_id
                        }
                        detections.append(detection)
        
        return detections
    
    def infer_image(self, image):
        """Run inference on single image"""
        # Preprocess
        processed_image = self.preprocess_image(image)
        
        # Run inference
        results = self.model(processed_image, verbose=False)
        
        # Postprocess
        detections = self.postprocess_results(results)
        
        return detections
    
    def infer_video_stream(self, camera_index=0, display=True):
        """Run inference on video stream"""
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_index}")
        
        # Set camera properties for optimal performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Starting video inference...")
        print("Press 'q' to quit, 's' to save current frame")
        
        frame_count = 0
        fps_start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break
            
            frame_count += 1
            
            # Run inference
            start_time = time.time()
            detections = self.infer_image(frame)
            inference_time = time.time() - start_time
            
            # Calculate FPS
            if frame_count % 30 == 0:
                current_time = time.time()
                fps = 30 / (current_time - fps_start_time)
                fps_start_time = current_time
                print(f"FPS: {fps:.1f}, Inference: {inference_time*1000:.1f}ms")
            
            # Draw results
            if display:
                annotated_frame = self.draw_detections(frame, detections)
                cv2.imshow('ExoGlove Detection', annotated_frame)
            
            # Print detections
            if detections:
                print(f"Frame {frame_count}: Found {len(detections)} objects")
                for det in detections:
                    print(f"  - {det['class']}: {det['confidence']:.2f}")
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame with detections
                if display and 'annotated_frame' in locals():
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"exoglove_detection_{timestamp}.jpg"
                    cv2.imwrite(filename, annotated_frame)
                    print(f"Saved frame: {filename}")
        
        cap.release()
        if display:
            cv2.destroyAllWindows()
    
    def draw_detections(self, image, detections):
        """Draw detection results on image"""
        annotated = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{det['class']}: {det['confidence']:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(annotated, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return annotated

def main():
    """Main function for Pi deployment"""
    # Model path - update this to your trained model
    model_path = "best.pt"  # Update this path
    
    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        print("Please update the model_path in the script")
        return
    
    # Initialize inference engine
    inference = ExoGloveInference(model_path, confidence_threshold=0.5)
    
    print("ExoGlove Inference Engine Ready!")
    print("Starting video stream inference...")
    
    try:
        # Start video inference
        inference.infer_video_stream(camera_index=0, display=True)
    except KeyboardInterrupt:
        print("\nInference stopped by user")
    except Exception as e:
        print(f"Error during inference: {e}")

if __name__ == "__main__":
    main()
