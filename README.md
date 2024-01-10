# Model
I trained Deep Learning Model(Yolov7) for CS2(new version of Counter Strike Global Offensive(CSGO)) , model classifies 4 classes : <br>
1) ct_body (body of ct side players)
2) ct_head (head of ct side players)
3) t_body (body of t side players)
4) t_head (head of t side players)

# Folders & Files
configuration_files : simply this folder contains configuration files of model <br>
model_history_curves : model's history graphs , precision recall ..<br>
test_images : after model created i test model with compeletly new images <br>
best_cs2_model.pt : Yolov7 model 

# Data
I collect dataset while playing death match with bots in CS2 (when i see bot in screen i took screenshot ) <br>
After collecting images I anotated them with Labelimg <br>

you can find dataset from my kaggle page , link : https://www.kaggle.com/datasets/merfarukgnaydn/counter-strike-2-body-and-head-classification <br><br>
dataset is small for now , i will increase it and train model again with new dataset<br>

# Some example Test Images

![a7](https://github.com/siromermer/CS2-Yolov7-Custom-ObjectDetection/assets/113242649/50026a12-b13c-4122-91d8-0c1b3f8ab79b)<br><br>
![a3](https://github.com/siromermer/CS2-Yolov7-Custom-ObjectDetection/assets/113242649/ed1836ef-c409-4cdc-b151-9da57f9a51bf)<br><br>
![a6](https://github.com/siromermer/CS2-Yolov7-Custom-ObjectDetection/assets/113242649/a9a09e1c-1ecd-491b-86c9-526ccd931415)



