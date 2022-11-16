import imghdr
from flask import Flask, render_template, request
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('index.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/predict',methods=['POST'])
def upload_image_file():
    model = load_model("models/mnistCNN.h5")
    if request.method == 'POST':
        img = Image.open(request.files['img']).convert('L')
        img = img.resize((28,28))
        im2arr = np.array(img)
        im2arr = im2arr.reshape(1,28,28,1)
        # predict = model.predict(im2arr)
        predict = model.predict([im2arr])[0]
        predicted =  np.argmax(predict)
        acc = max(predict)
        print(predicted,acc)
    
    return render_template('result.html',prediction=predicted,Accuracy=str(int(acc*100))+'%')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)
