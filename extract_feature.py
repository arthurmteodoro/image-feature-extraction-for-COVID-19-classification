import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def extract_feature(data_dir, model, process_image, verbose=1, validation_split=None):
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

    for each in classes:  # Loop for the folders
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
        j = j + 1

    if verbose == 1:
        pbar.close()

    np_batch = np.array(batch)
    np_labels = np.array(labels)
    np_images = np.array(images)

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


if __name__ == '__main__':
    model = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=True)
    layer_name = 'fc2'
    intermediate_layer_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    intermediate_layer_model.summary()

    input_train, input_test = extract_feature('/home/arthur/Programs/covid-ct', intermediate_layer_model, process_img,
                                              verbose=1, validation_split=0.2)

    from xDNN_class import *
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import cohen_kappa_score
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import confusion_matrix
    import time

    start = time.time()
    Output1 = xDNN(input_train, 'Learning')
    end = time.time()
    print("###################### Model Trained ####################")
    print("Time: ", round(end - start, 2), "seconds")

    input_test['xDNNParms'] = Output1['xDNNParms']

    startValidation = time.time()
    Mode2 = 'Validation'
    Output2 = xDNN(input_test, Mode2)
    endValidation = time.time()

    print("###################### Results ##########################")

    y_test_labels = input_test['Labels']

    # Elapsed Time
    print("Time: ", round(endValidation - startValidation, 2), "seconds")
    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(y_test_labels, Output2['EstLabs'])
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(y_test_labels, Output2['EstLabs'], average='micro')
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(y_test_labels, Output2['EstLabs'], average='micro')
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_test_labels, Output2['EstLabs'], average='micro')
    print('F1 score: %f' % f1)
    # kappa
    kappa = cohen_kappa_score(y_test_labels, Output2['EstLabs'])
    print('Cohens kappa: %f' % kappa)

    # confusion matrix
    matrix = confusion_matrix(y_test_labels, Output2['EstLabs'])
    print("Confusion Matrix: ", matrix)
