import tensorflow as tf
import numpy as np
import efficientnet.tfkeras as efns


def VGG16():
    base_model = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=True)
    intermediate_layer_model = tf.keras.Model(inputs=base_model.input,
                                              outputs=base_model.get_layer('fc2').output)
    return intermediate_layer_model


def VGG16_baseline():
    base_model = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=True)
    intermediate_layer_model = tf.keras.Model(inputs=base_model.input,
                                              outputs=base_model.get_layer('flatten').output)
    return intermediate_layer_model


def process_vgg16(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.vgg16.preprocess_input(x)
    return x


def EfficientNetB0():
    base_model = efns.EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = tf.keras.layers.Flatten()(base_model.get_layer('block6d_add').output)
    return tf.keras.Model(inputs=base_model.input, outputs=x)


def EfficientNetB0_baseline():
    base_model = efns.EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    return base_model


def process_efficientnetb0(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = efns.preprocess_input(x)
    return x


def VGG19():
    base_model = tf.keras.applications.vgg19.VGG19(weights='imagenet', include_top=True)
    intermediate_layer_model = tf.keras.Model(inputs=base_model.input,
                                              outputs=base_model.get_layer('fc2').output)
    return intermediate_layer_model


def VGG19_baseline():
    base_model = tf.keras.applications.vgg19.VGG19(weights='imagenet', include_top=True)
    intermediate_layer_model = tf.keras.Model(inputs=base_model.input,
                                              outputs=base_model.get_layer('flatten').output)
    return intermediate_layer_model


def process_vgg19(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.vgg19.preprocess_input(x)
    return x


def InceptionV3():
    base_model = tf.keras.applications.inception_v3.InceptionV3(weights='imagenet', include_top=True)
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.get_layer('mixed9').output)
    return tf.keras.Model(inputs=base_model.input, outputs=x)


def InceptionV3_baseline():
    base_model = tf.keras.applications.inception_v3.InceptionV3(weights='imagenet', include_top=True)
    intermediate_layer_model = tf.keras.Model(inputs=base_model.input,
                                              outputs=base_model.get_layer('avg_pool').output)
    return intermediate_layer_model


def process_inception_v3(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(299, 299))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.inception_v3.preprocess_input(x)
    return x
