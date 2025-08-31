 

# About Project
This project aims to detect enemy players inside the game in real-time and point the player’s aim directly to the enemy’s head to fire the weapon.
I collected the dataset inside the game by taking screenshots from time to time while playing.
After collecting images, annotation was done with LabelImg.
You can find the dataset on my Kaggle page, link: https://www.kaggle.com/datasets/merfarukgnaydn/counter-strike-2-body-and-head-classification

<br>
The dataset is small for now, but the results are really good. Lots of people gave me feedback about the data quality, and they were all positive because the images are from the game, not from real life. Therefore, there aren’t any differences in environments, there aren’t any sharp lighting changes between images, or shape differences in objects.
The `yolov8_csgo_cs2_model.pt` file uses the YOLOv8 nano model as its pretrained backbone for this project.
<br>

<br>

For more project, you can check my personal blog website: https://visionbrick.com/




# Demo video


 


https://github.com/siromermer/CS2-Yolov7-Custom-ObjectDetection/assets/113242649/69525835-9c82-40b9-acf8-678272df9490


<br> <br>

# Why not CS2 ?
there is no  Raw input and  Mouse acceleration options in cs2 , therefore even models can detect players without problem in cs2 there is problem with mouse movements . As soon as they add this setting options to CS2 , this will work without problem for sure

## Repository Structure
- `yolov8_csgo_cs2_model.pt` - Trained YOLOv8 model weights
- `Yolov8_cs2_csgo_demo.py` - Real-time detection and auto-aim demo script
- `model-training.ipynb` - Training notebook with model training process
- `model-results/` - Training results and model evaluation metrics
- `test_images/` - Sample test images for demonstration
- `configuration_files/` - YAML configuration files for training
- `requirements.txt` - Python dependencies with exact versions

## Quick Start
1. **Install PyTorch with CUDA support** for optimal performance: [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
2. Install dependencies: `pip install -r requirements.txt`
3. Run the demo: `python Yolov8_cs2_csgo_demo.py`
4. Press 'q' to exit








