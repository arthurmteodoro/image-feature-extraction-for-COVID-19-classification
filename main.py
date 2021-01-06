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
from sklearn.neighbors import KNeighborsClassifier

classifiers_available = [
    'xdnn',
    'knn'
]


def extract_features(data_dir, model, validation_dir, validation_split):
    model, process_img = feature_extractor.get_feature_extractor(model)
    model.summary()

    print('Extract Features from dataset')
    if validation_dir is None:
        input_train, input_test = extract_feature(data_dir, model, process_img,
                                                  verbose=1, validation_split=validation_split)
    else:
        input_train = extract_feature(data_dir, model, process_img, verbose=1, validation_split=None)
        input_test = extract_feature(validation_dir, model, process_img, verbose=1, validation_split=None)

    return input_train, input_test


def test(y_true, y_pred):
    print('Results')
    accuracy = accuracy_score(y_true, y_pred)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(y_true, y_pred)
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(y_true, y_pred)
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_true, y_pred)
    print('F1 score: %f' % f1)
    # kappa
    kappa = cohen_kappa_score(y_true, y_pred)
    print('Cohens kappa: %f' % kappa)
    # roc auc
    roc_auc = roc_auc_score(y_true, y_pred)
    print('ROC AUC: %f' % roc_auc)
    # confusion matrix
    matrix = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix: \n", matrix)


def run_xdnn(input_train, input_test):
    print('Training xDNN')
    output_train = xDNN(input_train, 'Learning')

    print('Validation xDNN')
    input_test['xDNNParms'] = output_train['xDNNParms']
    output_test = xDNN(input_test, 'Validation')

    return output_test['EstLabs']


def run_knn(input_train, input_test):
    print('Training KNN')
    clf = KNeighborsClassifier(15)
    clf.fit(input_train['Features'], input_train['Labels'])

    print('Validation KNN')
    y_pred = clf.predict(input_test['Features'])

    return y_pred


def run(data_dir, model, classifier, validation_dir, validation_split):
    input_train, input_test = extract_features(data_dir, model, validation_dir, validation_split)

    if classifier == 'xdnn':
        y_pred = run_xdnn(input_train, input_test)
    elif classifier == 'knn':
        y_pred = run_knn(input_train, input_test)

    test(input_test['Labels'], y_pred)


def main():
    parser = argparse.ArgumentParser(
        description='Train xDNN with CNN Feature Extractor'
    )
    parser.add_argument('--data-dir', help='Training dataset path', required=True)
    parser.add_argument('--model', help='Select model to feature extractor', required=True)
    parser.add_argument('--classifier', help='Select the Classifier method', required=True)
    parser.add_argument('--validation-dir', help='Validation dataset path', default=None)
    parser.add_argument('--validation-split', help='Percentage dataset to validation', default=0.2)

    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        print('Error: dataset root directory does not exist.')
        sys.exit(1)

    if args.model not in feature_extractor.avaliable_models:
        print('Error: --model value must be one of: ', ', '.join(feature_extractor.avaliable_models))
        sys.exit(1)

    if args.classifier not in classifiers_available:
        print('Error: --classifier value must be one of: ', ', '.join(classifiers_available))
        sys.exit(1)

    run(**vars(args))


if __name__ == '__main__':
    main()
