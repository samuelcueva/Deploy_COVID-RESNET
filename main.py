from flask import Flask, render_template, request,jsonify
import tensorflow as tf
from model import preprocess_image
import numpy as np
import os
from download_model import download,extract


IMAGE_SIZE = (256, 256)
PATHOLOGIES = ['COVID-19', 'NORMAL', 'PNEUMONIA']
path_img = 'images/image'

# parameters for Google Cloud Storage API
path_model = 'covid_resnet_model'
bucket_name = os.environ.get('BUCKET_NAME')
blob = 'covid_resnet_model.zip'
destination = 'model.zip'

# to download the serialized model
download(bucket_name,blob,destination)
# to extract the zip file
extract(destination)
# To load the serialized model
model = tf.keras.models.load_model(path_model)

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
    p = np.around(output[np.argmax(output)] * 100, decimals=4)

    return jsonify({'prediction':prediction,'percentage':p})
    # return


if __name__ == '__main__':
    app.run(port=5000, debug=False)
