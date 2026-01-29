 # About Project
Real-time  detection system with automated mouse control that detects and locks onto targets in milliseconds, reacting faster than human reflexes using YOLO and TensorRT for maximum inference performance.

<br>

I collected the dataset inside the game by taking screenshots from time to time while playing.
After collecting images, annotation was done with LabelImg.
You can find the dataset on my Kaggle page,  [link](https://www.kaggle.com/datasets/merfarukgnaydn/counter-strike-2-body-and-head-classification)
. The dataset is small for now, but the results are really good. Lots of people gave me feedback about the data quality, and they were all positive because the images are from the game, not from real life. Therefore, there aren’t any differences in environments, there aren’t any sharp lighting changes between images, or shape differences in objects.

<br>

The `yolov8_csgo_cs2_model.pt` file uses the YOLOv8 nano model as its pretrained backbone for this project. TensorRT engine files(`yolov8_csgo_cs2_model.engine`) must be exported locally as they are GPU-specific and optimized for your hardware.
<br>


For more project, you can check my personal blog website: https://visionbrick.com/


# Demo video

https://github.com/siromermer/CS2-Yolov7-Custom-ObjectDetection/assets/113242649/69525835-9c82-40b9-acf8-678272df9490


<br> 

# Disclaimer
This project is for educational and research purposes only. It should not be used for malicious intentions. The project only supports Counter-Strike: Global Offensive which is no longer playable online since Counter-Strike 2 was released.

# Why not CS2 ?
there is no  Raw input and  Mouse acceleration options in cs2 , therefore even models can detect players without problem in cs2 there is problem with mouse movements . As soon as they add this setting options to CS2 , this will work without problem for sure. TensorRT version is implemented for better performance.

## Repository Structure
- `yolov8_csgo_cs2_model.pt` - Trained YOLOv8 model weights
- `csgo_extension.py` - Real-time detection and auto-aim demo script
- `csgo_extension_tensorrt.py` - TensorRT accelerated demo script
- `export_tensorrt.py` - Export PyTorch model to TensorRT engine format
- `model-training.ipynb` - Training notebook with model training process
- `model-results/` - Training results and model evaluation metrics
- `test_images/` - Sample test images for demonstration
- `configuration_files/` - YAML configuration files for training
- `requirements.txt` - Python dependencies for standard PyTorch version
- `requirements-tensorrt.txt` - Python dependencies for TensorRT version

## Quick Start

### Standard Version
Requires Python 3.9+ with PyTorch CUDA support
1. Install PyTorch with CUDA support: [PyTorch Installation Guide](https://visionbrick.com/installation-guide-for-a-gpu-supported-pytorch-environment/)
2. Install dependencies: `pip install -r requirements.txt`
3. Run the demo: `python csgo_extension.py`
4. Press 'q' to exit

### TensorRT Version
Requires Python 3.9+ with PyTorch CUDA support and TensorRT
1. Install PyTorch with CUDA support: [PyTorch Installation Guide](https://visionbrick.com/installation-guide-for-a-gpu-supported-pytorch-environment/)
2. Install dependencies: `pip install -r requirements-tensorrt.txt`
3. Export model to TensorRT: `python export_tensorrt.py`
4. Run TensorRT demo: `python csgo_extension_tensorrt.py`
5. Press 'q' to exit










