import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD

from pathlib import Path
import csv
import numpy as np
import cv2
import sklearn
import keras
from sklearn.model_selection import train_test_split
from keras.layers import Dense,Flatten,MaxPooling2D,Dropout
from keras.layers.convolutional import Convolution2D
import matplotlib.image as mpimg
from keras import backend as K
from keras.utils import np_utils

# Generate dummy data

data = []
with open('./new_lables.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
          data.append([row[0],int(row[-1])])

train_samples, validation_samples = train_test_split(data, test_size=0.2)

def preprocess(image_list):
    not_match = []
    images=[]
    labels=[]
    for image_lable in image_list:                  
        image = mpimg.imread("./PlantVillage-Dataset/raw/color/"+image_lable[0])
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.resize(image, (32, 32))
        image = np.reshape(image, (32,32,1)) 
        images.append(image)
        labels.append(image_lable[1])
    
    return images,labels

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_images,batch_lables= preprocess(samples[offset:offset+batch_size])
            x_train=np.asfarray(batch_images)
            #x_train=keras.utils.to_categorical(x_train, num_classes=32)
            y_train=np.asfarray(batch_lables)
            y_train=keras.utils.to_categorical(y_train, num_classes=39)
            yield (x_train, y_train)


# In[ ]:


train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

model = Sequential()
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)))
model.add(Convolution2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(39, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=2)
model.save('model.h5')
#score = model.evaluate(x_test, y_test, batch_size=32)