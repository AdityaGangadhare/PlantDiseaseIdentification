from flask import Flask, request, send_from_directory
import base64
import time
import os
import numpy as np
import os
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
        print("images//" + image_name+"_"+tstamp + ".jpg")
        file.save("images//" + image_name + ".jpg")
        model = load_model("model.h5")
        image=mpimg.imread("images//" + image_name+".jpg")
        image = cv2.resize(image, (224, 224))
        #image = image.reshape(1,150528)
        image = np.reshape(image, (1,224, 224, 3))
        img_class = model.predict(image)
        image_class = np.argmax(img_class, axis=1)

    except Exception as e:
        print(e) 
        return "Error Occured", 400
    # return "Success Image Class --> ", 200
    return "Success Image Class --> "+str(image_class), 200


if __name__ == "__main__":
    app.run(host='localhost', port=5005, debug=True)
