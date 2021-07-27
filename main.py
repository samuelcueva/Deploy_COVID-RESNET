from flask import Flask, render_template, request,jsonify
import tensorflow as tf
from model import preprocess_image,create_covidResnet
import numpy as np
import os
from download_model import download,extract


IMAGE_SIZE = (256, 256)
PATHOLOGIES = ['COVID-19', 'NORMAL', 'PNEUMONIA']
path_img = 'images/image'

# parameters for Google Cloud Storage API 
bucket_name = os.environ.get('BUCKET_NAME')
blob = 'weights_covid_resnet.zip'
destination = '/tmp'
download_to = os.path.join(destination,blob)
path_weights = os.path.join(destination,'weights_covidResnet/cp-0030.ckpt')

# to download the trained weights(serialized) 
download(bucket_name,blob,download_to)

# to extract the zip file
extract(download_to,destination)

# create the base model(Resnet 50)
resnet50 = tf.keras.applications.ResNet50(include_top=False, input_shape=(256, 256, 3),
                                              weights=None)
# create COVID-RESNET
covid_resnet = create_covidResnet(resnet50)

# Load the trained weights(serialized) into the model
covid_resnet.load_weights(path_weights)


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # print(type(request.files['image']))
    request.files['image'].save(
        path_img)  # this save the image in a path but in production maybe it's not the better
    output = covid_resnet.predict(preprocess_image(path_img, IMAGE_SIZE))[0]

    prediction = PATHOLOGIES[np.argmax(output)]
    p = output[np.argmax(output)] * 100

    return jsonify({'prediction':prediction,'percentage':p})
    


if __name__ == '__main__':
    app.run(port=5000, debug=False)
