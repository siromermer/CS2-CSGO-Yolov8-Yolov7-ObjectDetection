 

# About Project
This project aims to detect enemy players inside the game in real-time and point the player’s aim directly to the enemy’s head to fire the weapon.
I collected the dataset inside the game by taking screenshots from time to time while playing.
After collecting images, annotation was done with LabelImg.
You can find the dataset on my Kaggle page, link: https://www.kaggle.com/datasets/merfarukgnaydn/counter-strike-2-body-and-head-classification

<br>
The dataset is small for now, but the results are really good. Lots of people gave me feedback about the data quality, and they were all positive because the images are from the game, not from real life. Therefore, there aren’t any differences in environments, there aren’t any sharp lighting changes between images, or shape differences in objects.
<br> <br>
Both YOLOv7 and YOLOv8 models are trained, and the best balance between real-time performance and accuracy was achieved by the YOLOv8m model. Depending on your GPU, for better FPS you can also train with the YOLOv8n model.
<br>

# Demo video


 


https://github.com/siromermer/CS2-Yolov7-Custom-ObjectDetection/assets/113242649/69525835-9c82-40b9-acf8-678272df9490


<br> <br>

# Why not CS2 ?
there is no  Raw input and  Mouse acceleration options in cs2 , therefore even models can detect players without problem in cs2 there is problem with mouse movements . As soon as they add this setting options to CS2 , this will work without problem for sure








