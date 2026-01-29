"""
Export CS2/CSGO YOLOv8 Model to TensorRT Engine Format

This script exports the trained CS2/CSGO YOLOv8 model to TensorRT engine format

Requirements:
- TensorRT must be installed (pip install tensorrt)
- CUDA-compatible NVIDIA GPU
- The trained model file: yolov8_csgo_cs2_model.pt

NOTE:   
The exported .engine file will be optimized specifically for your GPU, this process might take some time.
"""

from ultralytics import YOLO

# Configuration
MODEL_PATH = "yolov8_csgo_cs2_model.pt"  # Input: PyTorch model
OUTPUT_NAME = "yolov8_csgo_cs2_model.engine"  # Output: TensorRT engine

# Load the trained PyTorch model
model = YOLO(MODEL_PATH)

""" 
- Export to TensorRT engine format
- format="engine" - Exports to TensorRT
- device=0 - Uses GPU 0 (builds for your specific GPU architecture)
- half=True - Enables FP16 precision 
"""
model.export(format="engine", device=0, half=True)

print(f"TensorRT engine saved as: {OUTPUT_NAME}")
 