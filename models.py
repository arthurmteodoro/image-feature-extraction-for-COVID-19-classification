import tensorflow as tf
import numpy as np
import efficientnet.tfkeras as efns


def VGG16():
    base_model = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=True)
    intermediate_layer_model = tf.keras.Model(inputs=base_model.input,
                                              outputs=base_model.get_layer('fc2').output)
    return base_model


def process_vgg16(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.vgg16.preprocess_input(x)
    return x
