from flask import Flask, request, send_from_directory
import base64
import time
import os

from keras.applications.vgg16 import VGG16
from keras.layers import merge, Input
import numpy as np
import os
from pathlib import Path
import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Dense, Activation, Flatten
from keras.layers import merge, Input
from keras.models import Model
from keras.utils import np_utils
import cv2
import keras
# set the "static" directory as the static folder.
# this will ensure that all the static files are under one folder
app = Flask(__name__, static_url_path='/static')


@app.route('/image', methods=['POST'])
def upload_file():
    image_name = request.form['image_name']
    print(image_name)
    print(request.files)
    if 'image' not in request.files or 'image_name' not in request.form:
        return "No file found", 404
    try:
        file = request.files['image']
        
        if not os.path.exists("images"):
            os.mkdir("images")
        tstamp = str(time.time())
        file.save("images//" + image_name+"_"+tstamp + ".jpg")
        image=mpimg.imread("images//" + image_name+"_"+tstamp + ".jpg")
        image = cv2.resize(image, (224, 224))
        #image = image.reshape(1,150528)
        image = np.reshape(image, (1,224, 224, 3))
        img_class = model.predict(image)
        image_class = np.argmax(img_class, axis=1)

    except Exception as e:
        print(e) 
        return "Error Occured", 400
    return "Success Image Class --> "+str(image_class), 200


def load_model():
    image_input = Input(shape=(224, 224, 3))
    model = VGG16(input_tensor=image_input, include_top=True,weights='imagenet')

    for layer in model.layers[:-1]:
        layer.trainable = False
    num_classes = 38

    last_layer = model.get_layer('fc2').output
    out = Dense(39, activation='softmax', name='output')(last_layer)
    custom_vgg_model2 = Model(image_input, out)
    custom_vgg_model2.load_weights("model.h5")

    return custom_vgg_model2

if __name__ == "__main__":
    model = load_model()
    app.run(host='192.168.1.219', port=5005, debug=True)
