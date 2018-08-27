
# coding: utf-8

# In[1]:


from pathlib import Path
import csv
import numpy as np
import cv2
import sklearn
import keras
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,Flatten,MaxPooling2D,Dropout
from keras.layers.convolutional import Convolution2D
import matplotlib.image as mpimg
from keras import backend as K
from keras.utils import np_utils

# In[2]:


data = []
with open('./Labels.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        data.append([row[0],int(row[-1])])
print(data[1])


# In[3]:


# sklearn.utils.shuffle(data)
train_samples, validation_samples = train_test_split(data, test_size=0.2)


# In[4]:


print(len(data))
dict = {1:"Apple_scab",2:"Black_rot",3:"Cedar_apple_rust",4:"healthy",5:"Powdery_mildew",6:"Cercospora_leaf_spot Gray_leaf_spot",7:"Common_rust_",8:"Northern_Leaf_Blight",9:"Esca_Black_Measles",10:"Leaf_blight_Isariopsis_Leaf_Spot",11:"Haunglongbing_Citrus_greening",12:"Bacterial_spot",13:"Early_blight",14:"Late_blight",15:"Leaf_scorch",16:"Leaf_Mold",17:"Septoria_leaf_spot",18:"Spider_mites Two-spotted_spider_mite",19:"Target_Spot",20:"Tomato_mosaic_virus",21:"Tomato_Yellow_Leaf_Curl_Virus",22:"Apple_Frogeye_Spot"}


# In[5]:


def preprocess(image_list):
    not_match = []
    images=[]
    labels=[]
    for image_lable in image_list:                  
        image = mpimg.imread(image_lable[0])
        image = cv2.resize(image, (64, 64))
        images.append(image)
        labels.append(image_lable[1])
    
    return images,labels


# In[6]:


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_images,batch_lables= preprocess(samples[offset:offset+batch_size])
            x_train=np.asfarray(batch_images)
            #x_train=keras.utils.to_categorical(x_train, num_classes=32)
            y_train=np.asfarray(batch_lables)
            y_train=keras.utils.to_categorical(y_train, num_classes=32)
            yield x_train, y_train


# In[ ]:


train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


"""Model"""

model = Sequential()
# model.add(Convolution2D(24, 5, 5,subsample=(2,2),activation="relu",input_shape=(64,64,3)))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Convolution2D(36, 5, 5,subsample=(2,2),activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Convolution2D(48, 5, 5,subsample=(2,2),activation='relu'))
# model.add(Flatten())
# model.add(Dense(96))
# model.add(Dense(64))
# model.add(Dense(32))

model.add(Convolution2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(64,64,3)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='softmax'))
#model.compile(loss='mse',optimizer='adam')
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=2)
model.save('model.h5')


# In[ ]:


print(dict[1])


# In[ ]:


for no, disease in dict.items():
    if disease == "Black rot":
        print(no)

