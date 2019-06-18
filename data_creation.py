import pickle
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
img_path="data/"
log_path="data\driving_log.csv"

log=pd.read_csv(log_path)
"""
center,left,right=log.iloc[0,0],log.iloc[0,1],log.iloc[0,2]
left=left[1:]
right=right[1:]
img=Image.open(img_path+str(center))

img.show()
img=Image.open(img_path+str(left))

img.show()

img=Image.open(img_path+str(right))

img.show()
"""

def data_loading(delta,log):
    features=[]
    labels=[]
    print
    for i in tqdm(range(len(log))):
        for j in range(3):
            ipath=log.iloc[i,j]
            if(j!=0):
                ipath=ipath[1:]
            ipath=img_path+ipath
            #print(ipath)
            img=plt.imread(ipath)
            #img=(cv2.cvtColor(img,cv2.COLOR_RGB2HSV))[:,:,1]
            img=cv2.resize((cv2.cvtColor(img,cv2.COLOR_RGB2HSV))[:,:,1],(40,40))
            features.append(img)
            if(j==0):
                labels.append(float(log.iloc[i,3]))
            elif j==1:
                labels.append(float(log.iloc[i,3])+delta)
            else:
                labels.append(float(log.iloc[i,3])-delta)
    return features,labels
features,labels=data_loading(0.2,log)

features=np.array(features).astype('float32')
labels=np.array(labels).astype('float32')
np.save("features_40x40.npy",features)
np.save("labels_40x40.npy",labels)