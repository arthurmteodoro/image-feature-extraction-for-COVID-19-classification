import tensorflow as tf
import numpy as np
from xDNN_class import *
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import argparse
import os
import sys
import feature_extractor
from extract_feature import extract_feature


def run(data_dir, model):
    model, process_img = feature_extractor.get_feature_extractor(model)

    print('Extract Features from dataset')
    input_train, input_test = extract_feature(data_dir, model, process_img,
                                              verbose=1, validation_split=0.2)
    print('Training xDNN')
    output_train = xDNN(input_train, 'Learning')

    print('Validation xDNN')
    input_test['xDNNParms'] = output_train['xDNNParms']
    output_test = xDNN(input_test, 'Validation')

    print('Results')
    y_test_labels = input_test['Labels']
    accuracy = accuracy_score(y_test_labels, output_test['EstLabs'])
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(y_test_labels, output_test['EstLabs'], average='micro')
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(y_test_labels, output_test['EstLabs'], average='micro')
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_test_labels, output_test['EstLabs'], average='micro')
    print('F1 score: %f' % f1)
    # kappa
    kappa = cohen_kappa_score(y_test_labels, output_test['EstLabs'])
    print('Cohens kappa: %f' % kappa)
    # roc auc
    roc_auc = roc_auc_score(y_test_labels, output_test['EstLabs'])
    print('ROC AUC: %f' % roc_auc)
    # confusion matrix
    matrix = confusion_matrix(y_test_labels, output_test['EstLabs'])
    print("Confusion Matrix: ", matrix)


def main():
    parser = argparse.ArgumentParser(
        description='Train xDNN with CNN Feature Extractor'
    )
    parser.add_argument('--data_dir', help='Training dataset path', required=True)
    parser.add_argument('--model', help='Select model to feature extractor', required=True)

    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        print('Error: dataset root directory does not exist.')
        sys.exit(1)

    if args.model not in feature_extractor.avaliable_models:
        print('Error: --model value must be one of: ', ', '.join(feature_extractor.avaliable_models))
        sys.exit(1)

    run(**vars(args))


if __name__ == '__main__':
    main()