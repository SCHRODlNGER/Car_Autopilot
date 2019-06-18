import numpy as np
from keras.layers import *
from keras.models import *
from keras.activations import *
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau

def keras_model():
    model=Sequential()
    model.add(Conv2D(32,(3,3),padding="same"))
    model.add(Conv2D(32,(3,3),padding="same"))
    model.add(MaxPool2D())
    model.add(Conv2D(64,(3,3),padding="same"))
    model.add(Conv2D(64,(3,3),padding="same"))
    model.add(MaxPool2D())
    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
   # model.add(relu())
    model.add(Dense(256,activation='relu'))
   # model.add(relu())
    model.add(Dense(128,activation='relu'))
   # model.add(relu())
    model.add(Dense(1))
    
    model.compile(optimizer="adam",loss="mse")
    
    filepath="selfdrivingv1.h5"
    
    checkpoint= ModelCheckpoint (filepath,verbose=1,save_best_only=True)
    lr=ReduceLROnPlateau(factor=0.1,patience=3,min_lr=1e-8)
    callbacks=[checkpoint,lr]
    return model,callbacks


features=np.load("features_40x40.npy")
labels=np.load("labels_40x40.npy")

#augment data

features=np.append(features,features[:,:,::-1],axis=0)
labels=np.append(labels,-labels,axis=0)
features=features.reshape(features.shape[0],40,40,1)
print(features.shape)

model,callbacks=keras_model()

from sklearn.model_selection import train_test_split as split
train_x,test_x,train_y,test_y=split(features,labels,test_size=0.1,random_state=1)
print(train_x[0])
model.fit(x=train_x,y=train_y,epochs=10,batch_size=64,callbacks=callbacks,validation_data=(test_x,test_y))

print(model.summary())
model.save("selfdriving1v1.h5")
