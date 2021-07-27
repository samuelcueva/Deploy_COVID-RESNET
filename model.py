import os
import tensorflow as tf

path_weights = 'weights_covidResnet/cp-0030.ckpt'
path_model = '../'


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


if __name__ == "__main__":
    # create the base model(Resnet 50)
    resnet50 = tf.keras.applications.ResNet50(include_top=False, input_shape=(256, 256, 3),
                                              weights=None)

    # create COVID-RESNET
    covid_resnet = create_covidResnet(resnet50)

    # Load the trained weights into the model
    covid_resnet.load_weights(path_weights)

    # Save the trained model
    covid_resnet.save(os.path.join(path_model,'covid_resnet_model.h5'))
