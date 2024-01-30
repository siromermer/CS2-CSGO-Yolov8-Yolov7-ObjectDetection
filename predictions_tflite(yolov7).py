import cv2
import numpy as np
import tensorflow as tf

"""
    this script used for tflite models which are converted from yolov7 models 
"""

image_path="test_images/a8.png"

"""
    money model classes : classes = ["bir","yirmibes","elli","on","bes",""]
    cs2 model classes :  ["ct_body","ct_head","t_body","t_head",""]

"""
classes = ["bir","yirmibes","elli","on","bes",""]
# classes = ["ct_body","ct_head","t_body","t_head",""]

# create interpreter
interpreter = tf.lite.Interpreter(model_path="cs2_yolo.tflite")

# it prepare input and output vector of models  
interpreter.allocate_tensors() 

image = cv2.imread(image_path)

# scale numbers 
org_w=image.shape[1] # take width 
org_h=image.shape[0] # take height

# scale output width coordinates with this numbers
r_w=float(org_w/640)

# scale output height coordinates with this numbers
r_h=float(org_h/640)

# copy image , dont make any operation to original image
copy_image=image.copy()

# convert copy_image BGR to RGB
copy_image = cv2.cvtColor(copy_image, cv2.COLOR_BGR2RGB)

# resize copy_image for model , every model accepts different size , in this case yolov7 model expects (640,640)
copy_image = cv2.resize(copy_image, (640, 640),interpolation=cv2.INTER_LINEAR)

"""
    OpenCV img = cv2.imread(path) loads an image with HWC-layout (height, width, channels), 
    while Pytorch requires CHW-layout. 
    So we have to do np.transpose(image,(2,0,1)) for HWC->CHW transformation.
"""
# transpose it for model , model expects  CHW layout 
copy_image = copy_image.transpose((2, 0, 1))

# add one more dimension 
copy_image = np.expand_dims(copy_image, 0)

# improve performance by ensuring that the data is stored in contiguous memory locations
copy_image = np.ascontiguousarray(copy_image)

# normalize image 
copy_image = copy_image.astype(np.float32)
copy_image /= 255


# get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Sets the value of the input tensor
interpreter.set_tensor(input_details[0]['index'], copy_image)


# call interpreter
interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])
output_data[0]
"""
    output data returns : batch_id , x1 , y1 , x2 ,y2 , class_id , score
    it returns for every possible object ,  filter them with score because it returns even scores which are very close to zero 
"""


# loop through all possible objects that are inside of output_data
for i,(_,x1,y1,x2,y2,class_id,score) in enumerate(output_data):
    if score > 0.5 :

        # create box array
        box = np.array([x1,y1,x2,y2])
        box = box.round().astype(np.int32).tolist()

        # round score values
        score = round(float(score),3)
        
        # take name with class_id from classes list
        name = classes[int(class_id)]
        
        name +=" "+str(score)

        # scale box coordinates , ( i find r_w and r_h when loading image)
        box[0]=int(box[0]*r_w)
        box[1]=int(box[1]*r_h)
        box[2]=int(box[2]*r_w)
        box[3]=int(box[3]*r_h)

        

        # use original image , remember i didnt touch it 
        cv2.rectangle(image , (box[0],box[1]) , (box[2],box[3]) , (0,255,0) , 2 )
        cv2.putText(image,name,(box[0], box[1] - 3),cv2.FONT_HERSHEY_SIMPLEX,0.75,[225, 255, 255],thickness=2)  


try:
    result_name=image_path.split("/")[1]
    result_name=result_name.split(".")[0]

    cv2.imwrite(f"results/{result_name}_result.jpg",image)
    
except:
    print("no image")




