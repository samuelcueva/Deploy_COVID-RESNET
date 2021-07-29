from flask import Flask, render_template, request,jsonify
import tensorflow as tf

from google.cloud import storage 
from zipfile import ZipFile

import numpy as np
import os



IMAGE_SIZE = (256, 256)
PATHOLOGIES = ['COVID-19', 'NORMAL', 'PNEUMONIA']
path_img = 'images/image'

def download(bucket_name,source_blob_name,destination):


    # bucket_name = "your-bucket-name"
    # source_blob_name = "storage-object-name"
    # destination = "local/path/to/file"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)

    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination)

    print(
        "Blob {} downloaded to {}.".format(
            source_blob_name, destination
        )
    )

def extract(file,destination):
    
    # opening the zip file in READ mode
    with ZipFile(file, 'r') as zip:

        # printing all the contents of the zip file
        zip.printdir()
  
        # extracting all the files
        print('Extracting all the files now...')
        zip.extractall(destination)
        #os.remove(file)
        print('Done!')


def preprocess_image(path, image_size):
    raw_img = tf.keras.preprocessing.image.load_img(path)
    img_array = tf.keras.preprocessing.image.img_to_array(raw_img)
    img = tf.keras.preprocessing.image.smart_resize(img_array, image_size)
    img = tf.expand_dims(img, 0)
    return img


def create_covidResnet(base_model):
    """create top layers to customize ResNet50
       to a network for Covid-19 detection"""

    input = tf.keras.Input(shape=(256, 256, 3))
    preprocess_input = tf.keras.applications.resnet.preprocess_input(input)
    model_base = base_model(preprocess_input)
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()(model_base)
    drop_out_1 = tf.keras.layers.Dropout(0.4)(global_average_layer)
    dense_layer = tf.keras.layers.Dense(4096, activation='relu')(drop_out_1)
    drop_out_2 = tf.keras.layers.Dropout(0.4)(dense_layer)
    output_model = tf.keras.layers.Dense(3, activation='softmax')(drop_out_2)

    # create the model
    model = tf.keras.models.Model(inputs=input, outputs=output_model)

    return model



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
