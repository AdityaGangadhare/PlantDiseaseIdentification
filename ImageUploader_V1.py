from flask import Flask, request, send_from_directory
import base64
import time
import os
import tensorflow as tf
os.environ["MKL_THREADING_LAYER"] = "GNU"
import numpy as np
from pathlib import Path
import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
from keras.models import load_model
import cv2
import keras

# set the "static" directory as the static folder.
# this will ensure that all the static files are under one folder
app = Flask(__name__)
modle = None

def load():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    model = load_model("model.h5")

def prepare_image(image):
    image = cv2.resize(image, (224, 224))
    #image = image.reshape(1,150528)
    image = np.reshape(image, (1,224, 224, 3))
    return image
    

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
        print("images//" + image_name+"_"+tstamp + ".jpg")
        file.save("images//" + image_name + ".jpg")
        
        image = mpimg.imread("images//" + image_name+".jpg")
        image = prepare_image(image)
        with graph.as_default():
            img_class = model.predict(image)
        image_class = np.argmax(img_class, axis=1)

    except Exception as e:
        print(e) 
        return "Error Occured", 400
    return "Success_"+str(image_class[0]), 200


if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    global model
    model = load_model("model.h5")
    graph = tf.get_default_graph()
    app.run(host='172.31.10.85', port=54853, debug=False)