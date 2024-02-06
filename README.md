# Updates !!!
In first version of this project , i trained yolov7 model and test it  only in images . Model was detecting nearly half of the players therefore i didnt test that model inside of real game because model was not that good . Therefore i trained one more model , this time i used Yolov8 model(YOLOv8m)  , when i compare those 2 model , yolov8 is outperforming yolov7 .  Yolov8m model is nearly detecting all players(even when players are dead and laying in ground D:) . So  with Yolov8m i decided to test it in real game and results was pretty good , check below video <br><br>Yolov8_cs2_csgo_demo.py --> use this file for testing model in game(use it with yoloV8 models)(it doesnt work with yolov7 model because ultralytics dont support yolov7 models)<br><br>
Note : left screen is original game screen , right screen is for seeing players heads with rectangle(not necessary screen you dont have to use this screen)

 


https://github.com/siromermer/CS2-Yolov7-Custom-ObjectDetection/assets/113242649/69525835-9c82-40b9-acf8-678272df9490



# Why not CS2 ?
there is no  Raw input and  Mouse acceleration options in cs2 , therefore even models can detect players without problem in cs2 there is problem with mouse movements . As soon as they add this setting options to CS2 , this will work without problem for syre


# Model
I trained Deep Learning Model(Yolov7) for CS2(new version of Counter Strike Global Offensive(CSGO)) , model classifies 4 classes : <br>
1) ct_body (body of ct side players)
2) ct_head (head of ct side players)
3) t_body (body of t side players)
4) t_head (head of t side players)


# Folders & Files
Yolov8_cs2_csgo_demo.py   :   main file for using Yolov8 model in game , this file doesnt work with yolov7 model because ultralytics dont support yoloV7 models (YOLOV8 FİLE) <br><br>
configuration_files : simply this folder contains configuration files of model (YOLOV7 FİLE) <br>  
model_history_curves : model's history graphs , precision recall .. (YOLOV7 FİLE) <br>  
test_images : after model created i test model with compeletly new images (YOLOV7 FİLE) <br>
best_cs2_model.pt : Yolov7 model (YOLOV7 FİLE)

# Data
I collect dataset while playing death match with bots in CS2 (when i see bot in screen i took screenshot ) <br>
After collecting images I anotated them with Labelimg <br>

you can find dataset from my kaggle page , link : https://www.kaggle.com/datasets/merfarukgnaydn/counter-strike-2-body-and-head-classification <br><br>
dataset is small for now , i will increase it and train model again with new dataset<br>





