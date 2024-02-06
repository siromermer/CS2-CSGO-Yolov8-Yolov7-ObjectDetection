
from mss import mss
from PIL import Image, ImageGrab

from ultralytics import YOLO
import time
import cv2 
import numpy as np
import time

import pyautogui

"""
    Raw input : off
    Mouse acceleration : off
    sensivity : 3.85
    screen_size : height=480 , width=640

"""

avg_fps=0

img = None
t0 = time.time()
n_frames = 1

model = YOLO('yolov8_100epoch.pt')

label_dict={1:"ct_body",2:"ct_head",3:"t_body",4:"t_head"}

#pyautogui.FAILSAFE=False

sct = mss()
while True:
    
    img = np.array(sct.grab((0,0,640,480)))    
 
    
    # img = cv.cvtColor(img, cv.COLOR_RGB2BGR)   
    #img=img[:,:,:3]
    region_of_interest = img[:480, :640 ,:3]   
    region_of_interest = np.ascontiguousarray(region_of_interest, dtype=np.uint8) # this solved my issue !!!!!!!!!!!
 
    
    # Run inference on the source
    results = model(region_of_interest)

    result_list=[]
    class_list=[]
    conf_list=[]

    k=0
    for result in results: 
        
        for class_name in result.boxes.cls:
            class_list.append(int(class_name))

        
        for id,box in enumerate(result.boxes.xyxy) : # box with xyxy format, (N, 4)
            if k==0:
                if label_dict[class_list[id]]=="t_head" or label_dict[class_list[id]]=="ct_head":
                    x1,y1,x2,y2=int(box[0]),int(box[1]),int(box[2]),int(box[3])
      
                    x_mid=int((x1+x2)/2)
                    y_mid=int((y1+y2)/2)
                    
                    pyautogui.moveTo(x_mid,y_mid)
                    pyautogui.click(x1+5,y1)
                    
                    cv2.rectangle(region_of_interest,(x1,y1),(x2,y2),(0,0,255),2)
                    cv2.putText(region_of_interest, str(avg_fps) , (20,50) , cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0), 1, cv2.LINE_AA)
                    k+=1
        
        


    #region_of_interest=cv2.cvtColor(region_of_interest,cv2.COLOR_RGB2BGR)
    cv2.imshow("Computer Vision", region_of_interest)

    # Break loop and end test
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    
    elapsed_time = time.time() - t0
    avg_fps = (n_frames / elapsed_time)
    print("Average FPS: " + str(avg_fps))
    #cv2.putText(region_of_interest, str(avg_fps) , (50,50) , cv2.FONT_HERSHEY_SIMPLEX,3,(255,0,0), 3, cv2.LINE_AA)
    n_frames += 1


