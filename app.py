#pip install flask tensorflow scipy opencv-contrib-python numpy

from flask import request, render_template, redirect, url_for ,Flask, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import scipy
import os

UPLOAD_FOLDER = "static/images/uploads"


classes = ["Cataract", "Crossed Eyes", "Glucoma", "Retinopathy", "Uveitis"]

app = Flask(__name__)
app.config['IMAGE_UPLOADS'] = UPLOAD_FOLDER


# Define the allowed file extensions that can be uploaded
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def pre(img_path):
    # Input image
    # try:
    cnn_model = load_model("CNN_Eye_Disease_Model.h5")
    cnn_model.summary()
    img = cv2.imread(img_path) 
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(100,100))
    # img = np.array(img)# Creating kernel 
    kernel = np.ones((1, 1), np.uint8) 
    eroded = cv2.erode(img,kernel)
    img = cv2.dilate(eroded,kernel)
    img = img.reshape((1,100,100,3))
    type_pred = cnn_model.predict(img)

    print(type_pred)

    # except Exception as e:
    #     print(e)
    
    return classes[type_pred.argmax()]


@app.route('/')
def image_upload():
    return render_template('image_upload.html', predictions=[])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if 'file' not in request.files:
        return render_template('image_upload.html', predictions=[])
    
    file = request.files['file']
    if file.filename == '':
        return render_template('image_upload.html', predictions=[])
    
    if file and allowed_file(file.filename):
        path = os.path.join(app.config['IMAGE_UPLOADS'], file.filename)
        file.save(path)
        print("path",path)
        predictions = pre(path)
        print(classes)
        print("Prediction:", predictions)

        if path.endswith(".dcm"):
            path = os.path.join(r"static/images", "no_preview.jpg")
            print(path)
        return render_template('image_upload.html', predictions=predictions, path=path)#.split("\\",1)[1])

    return render_template('image_upload.html', predictions=[])


if __name__ == "__main__":
        app.run(debug=True, port=8081)

