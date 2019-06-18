import numpy as np
import cv2
from keras.models import *
from keras.layers import *

def keras_predict(model,image):
    processed= process_image(image)
    steering_angle= float(model.predict(processed,batch_size=1))
    steering_angle= steering_angle*200
    return steering_angle

def process_image(image):
    image_x=40
    image_y=40
    img=cv2.resize(image,(image_x,image_y))
    img=np.reshape(img,(-1,image_x,image_y,1))
    return img

smoothed_angle=0
model=load_model("selfdriving1v1.h5")

wheel=cv2.imread('swheel.jpg',0)
rows,cols=wheel.shape

cap=cv2.VideoCapture('run.mp4')
print(cap.isOpened())
while(cap.isOpened()):
    ret,frame=cap.read()
    gray=cv2.resize((cv2.cvtColor(frame,cv2.COLOR_RGB2HSV))[:,:,1],(40,40))
    
    steering_angle=keras_predict(model,gray)
    smoothed_angle += 0.2 * pow(abs((steering_angle - smoothed_angle)), 2.0 / 3.0) * (
        steering_angle - smoothed_angle) / abs(
        steering_angle - smoothed_angle)
    print(smoothed_angle)
    m=cv2.getRotationMatrix2D((cols/2,rows/2),-smoothed_angle,1)
    dst=cv2.warpAffine(wheel,m,(cols,rows))
    cv2.imshow("steering wheel",dst)
    cv2.imshow('frame',cv2.resize(frame,(500,300),interpolation=cv2.INTER_AREA))
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break

cap.release()
cv2.destroyAllWindows()