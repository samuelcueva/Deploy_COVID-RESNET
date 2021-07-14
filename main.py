from flask import Flask, render_template, request,jsonify
import tensorflow as tf
from model import preprocess_image
import numpy as np

IMAGE_SIZE = (256, 256)
PATHOLOGIES = ['COVID-19', 'NORMAL', 'PNEUMONIA']
path_img = 'images/image.jpg'
# To load the serialized model
model = tf.keras.models.load_model('covid_resnet_model')

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # print(type(request.files['image']))
    request.files['image'].save(
        path_img)  # this save the image in a path but in production maybe it's not the better
    output = model.predict(preprocess_image(path_img, IMAGE_SIZE))[0]

    prediction = PATHOLOGIES[np.argmax(output)]
    p = output[np.argmax(output)] * 100

    return jsonify({'prediction':prediction,'percentage':p})
    # return


if __name__ == '__main__':
    app.run(port=5000, debug=True)
