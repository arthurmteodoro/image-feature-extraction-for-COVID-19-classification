import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import random
import sys


def process_ds_extract_name_label(filename, classes):
    class_name = tf.strings.split(filename, '/')[-2]
    class_index = tf.where(tf.equal(classes, class_name))[:, -1][0]
    return filename, class_index


def process_ds_load_image(filename, process_img_fn):
    image = process_img_fn(filename)
    return image


def extract_feature(data_dir, model, process_image, verbose=1, validation_split=None, shuffle=True):
    # check if data_dir is a dir
    if not (os.path.isdir(data_dir)):
        raise Exception('Data Dir is not a directory')

    contents = os.listdir(data_dir)
    classes = [each for each in contents if os.path.isdir(os.path.join(data_dir, each))]

    images = []
    batch = []
    labels = []

    j = 0

    if verbose == 1:
        total_imgs = 0
        for each in classes:
            class_path = os.path.join(data_dir, each)
            total_imgs += len(os.listdir(class_path))

        pbar = tqdm(total=total_imgs, unit='imgs')

    files = []
    for each in classes:  # Loop for the folders
        class_path = os.path.join(data_dir, each)
        files_class = os.listdir(class_path)
        images_full_path = [os.path.abspath(os.path.join(class_path, x)) for x in files_class]
        files.extend(images_full_path)

    ds_files = tf.data.Dataset.list_files(files, shuffle=False)

    classes_tensor = tf.convert_to_tensor(classes)
    classes_tensor = list(classes_tensor.numpy())

    AUTO = tf.data.experimental.AUTOTUNE

    ds_name_image = ds_files.map(lambda filename: process_ds_extract_name_label(filename, classes_tensor),
                                 num_parallel_calls=AUTO)
    # ds_name_image = ds_name_image.batch(8)

    ds_image = ds_files.map(lambda filename: process_ds_load_image(filename, process_image), num_parallel_calls=AUTO)
    # ds_image = ds_image.batch(8)
    ds_image.prefetch(200)

    i = 0
    for name_image, img in zip(iter(ds_name_image), iter(ds_image)):
        img_name, class_index = name_image

        img_name = str(img_name.numpy(), 'utf-8')
        class_index = class_index.numpy()

        features = model.predict(img)
        # Append features and labels
        batch.append(features[0])
        images.append(img_name)
        labels.append(class_index)
        if verbose == 1:
            pbar.update(1)

    sys.exit(0)

    '''for each in classes:  # Loop for the folders
        class_path = os.path.join(data_dir, each)
        files = os.listdir(class_path)

        for ii, file in enumerate(files, 1):  # Loop for the imgs inside the folders
            # load images from file path
            x = process_image(os.path.join(class_path, file))
            # Extract features
            features = model.predict(x)
            # Append features and labels
            batch.append(features[0])
            images.append(file)
            labels.append(str(j))
            if verbose == 1:
                pbar.update(1)
        j = j + 1'''

    if verbose == 1:
        pbar.close()

    np_batch = np.array(batch).astype(np.float64)
    np_labels = np.array(labels)
    np_images = np.array(images)

    if shuffle:
        indices = np.arange(np_batch.shape[0])
        np.random.shuffle(indices)

        np_batch = np_batch[indices]
        np_labels = np_labels[indices]
        np_images = np_images[indices]

    if validation_split is not None:
        np_labels_t = np_labels.reshape(-1, 1)
        np_images_t = np_images.reshape(-1, 1)

        np_images_labels = np.hstack((np_images_t, np_labels_t))

        x_train, x_test, y_train, y_test = train_test_split(np_batch, np_images_labels,
                                                            test_size=validation_split)

        split_train = np.hsplit(y_train, 2)
        y_train_images = split_train[0].reshape((split_train[0].shape[0],))
        y_train_labels = split_train[1].reshape((split_train[1].shape[0],)).astype(np.int32)

        split_test = np.hsplit(y_test, 2)
        y_test_images = split_test[0].reshape((split_test[0].shape[0],))
        y_test_labels = split_test[1].reshape((split_test[1].shape[0],)).astype(np.int32)

        input_train = {'Images': y_train_images, 'Features': x_train, 'Labels': y_train_labels}
        input_test = {'Images': y_test_images, 'Features': x_test, 'Labels': y_test_labels}

        return input_train, input_test
    else:
        input_train = {'Images': np_images, 'Features': np_batch, 'Labels': np_labels.astype(np.int32)}
        return input_train


def process_img(path):
    img = tf.keras.preprocessing.image.load_img(path, target_size=(224, 224))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.vgg16.preprocess_input(x)
    return x
