import numpy as np
import tensorflow as tf
import os
import pickle

path_weights='weights_covidResnet/cp-0030.ckpt'
path_images='resnet_images'

IMAGE_SIZE=(256,256)
LR=0.01
PATHOLOGIES=['COVID-19','NORMAL','PNEUMONIA']

def preprocess_image(path,image_size):
    raw_img=tf.keras.preprocessing.image.load_img(path)
    img_array=tf.keras.preprocessing.image.img_to_array(raw_img)
    img=tf.keras.preprocessing.image.smart_resize(img_array,image_size)
    img=tf.expand_dims(img,0)
    return img

def create_covidResnet(base_model):
  """create top layers to customize ResNet50
     to a network for Covid-19 detection"""

  input=tf.keras.Input(shape=(256,256,3))
  preprocess_input=tf.keras.applications.resnet.preprocess_input(input)
  model_base=base_model(preprocess_input)
  global_average_layer=tf.keras.layers.GlobalAveragePooling2D()(model_base)
  drop_out_1=tf.keras.layers.Dropout(0.4)(global_average_layer)
  dense_layer=tf.keras.layers.Dense(4096,activation='relu')(drop_out_1)
  drop_out_2=tf.keras.layers.Dropout(0.4)(dense_layer)
  output_model=tf.keras.layers.Dense(3,activation='softmax')(drop_out_2)

  #create the model
  model=tf.keras.models.Model( inputs=input,outputs=output_model)

  return model

# create the base model(Resnet 50)
resnet50=tf.keras.applications.ResNet50( include_top=False,input_shape=(256,256,3) ,weights=None)

# create COVID-RESNET
covid_resnet=create_covidResnet(resnet50)

# Load the trained weights into the model
covid_resnet.load_weights('weights_covidResnet/cp-0030.ckpt')

# Save the trained model
covid_resnet.save('covid_resnet_model')


restored_model = tf.keras.models.load_model('covid_resnet_model')



output=restored_model.predict(preprocess_image(os.path.join(path_images,'covid_2.jpg'),IMAGE_SIZE))

prediction=PATHOLOGIES[np.argmax(output)]

print(prediction)