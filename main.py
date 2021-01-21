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
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelBinarizer

classifiers_available = [
    'xdnn',
    'knn',
    'svm',
    'svm-linear'
]


def extract_features(data_dir, model, validation_dir, validation_split, full_ds=False):
    model, process_img = feature_extractor.get_feature_extractor(model)
    # model.summary()

    print('Extract Features from dataset')
    if validation_dir is None:
        input_train, input_test = extract_feature(data_dir, model, process_img,
                                                  verbose=1, validation_split=validation_split)
    elif full_ds:
        input_train = extract_feature(data_dir, model, process_img, verbose=1, validation_split=None)
        input_test = None
    else:
        input_train = extract_feature(data_dir, model, process_img, verbose=1, validation_split=None)
        input_test = extract_feature(validation_dir, model, process_img, verbose=1,
                                     validation_split=None, shuffle=False)

    return input_train, input_test


def test(y_true, y_pred):

    def multiclass_roc_auc_score(y_true, y_pred, average='macro'):
        lb = LabelBinarizer()
        lb.fit(y_true)
        Y_true = lb.transform(y_true)
        Y_pred = lb.transform(y_pred)

        return roc_auc_score(Y_true, Y_pred, average=average)

    accuracy = accuracy_score(y_true, y_pred)
    # precision tp / (tp + fp)
    precision = precision_score(y_true, y_pred, average='macro')
    # recall: tp / (tp + fn)
    recall = recall_score(y_true, y_pred, average='macro')
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_true, y_pred, average='macro')
    # kappa
    kappa = cohen_kappa_score(y_true, y_pred)
    # roc auc
    roc_auc = multiclass_roc_auc_score(y_true, y_pred, average='macro')
    # confusion matrix
    matrix = confusion_matrix(y_true, y_pred)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'kappa': kappa,
        'roc_auc': roc_auc,
        'matrix': matrix
    }


def print_results(accuracy, precision, recall, f1, kappa, roc_auc, matrix):
    print('Results')
    print('Accuracy: %f' % accuracy)
    print('Precision: %f' % precision)
    print('Recall: %f' % recall)
    print('F1 score: %f' % f1)
    print('Cohens kappa: %f' % kappa)
    print('ROC AUC: %f' % roc_auc)
    print("Confusion Matrix: \n", matrix)


def run_xdnn(input_train, input_test):
    print('Training xDNN')
    output_train = xDNN(input_train, 'Learning')

    print('Validation xDNN')
    input_test['xDNNParms'] = output_train['xDNNParms']
    output_test = xDNN(input_test, 'Validation')

    return output_test['EstLabs']


def run_knn(input_train, input_test, n_neighbors):
    print('Training KNN')
    clf = KNeighborsClassifier(n_neighbors)
    clf.fit(input_train['Features'], input_train['Labels'])

    print('Validation KNN')
    y_pred = clf.predict(input_test['Features'])

    return y_pred


def run_svm(input_train, input_test, kernel):
    print('Training SVM')
    if kernel == 'linear':
        clf = SVC(kernel='linear', C=1000, probability=True, class_weight='balanced')
    else:
        clf = SVC(gamma=1e-5, C=1000, probability=True, class_weight='balanced')

    clf.fit(input_train['Features'], input_train['Labels'])

    print('Validation SVM')
    y_pred = clf.predict(input_test['Features'])

    return y_pred


def run_training(input_train, input_test, classifier, n_neighbors):
    if classifier == 'xdnn':
        y_pred = run_xdnn(input_train, input_test)
    elif classifier == 'knn':
        y_pred = run_knn(input_train, input_test, n_neighbors)
    elif classifier == 'svm-linear':
        y_pred = run_svm(input_train, input_test, 'linear')
    elif classifier == 'svm':
        y_pred = run_svm(input_train, input_test, None)

    return y_pred


def run_one_shot(data_dir, model, classifier, validation_dir, validation_split, n_neighbors):
    input_train, input_test = extract_features(data_dir, model, validation_dir, validation_split)

    y_pred = run_training(input_train, input_test, classifier, n_neighbors)

    results = test(input_test['Labels'], y_pred)
    print_results(**results)


def run_cv(data_dir, model, classifier, validation_dir, validation_split, cv, n_neighbors):
    acc = []
    precision = []
    recall = []
    f1 = []
    kappa = []
    auc = []

    input_train, _ = extract_features(data_dir, model, validation_dir, validation_split, full_ds=True)

    skf = StratifiedKFold(n_splits=cv)

    for index, (train_index, test_index) in enumerate(skf.split(input_train['Features'], input_train['Labels'])):
        fold_input_train = {
            'Features': input_train['Features'][train_index],
            'Images': input_train['Images'][train_index],
            'Labels': input_train['Labels'][train_index]
        }

        fold_input_test = {
            'Features': input_train['Features'][test_index],
            'Images': input_train['Images'][test_index],
            'Labels': input_train['Labels'][test_index]
        }

        y_pred = run_training(fold_input_train, fold_input_test, classifier, n_neighbors)

        results = test(fold_input_test['Labels'], y_pred)
        print('Results for fold %d' % (index+1, ))
        print_results(**results)

        acc.append(results['accuracy'])
        precision.append(results['precision'])
        recall.append(results['recall'])
        f1.append(results['f1'])
        kappa.append(results['kappa'])
        auc.append(results['roc_auc'])

    print('----------------------- Final results --------------------------')
    print('Accuracy: %f +/- %f' % (np.mean(acc), np.std(acc)))
    print('Precision: %f +/- %f' % (np.mean(precision), np.std(precision)))
    print('Recall: %f +/- %f' % (np.mean(recall), np.std(recall)))
    print('F1 score: %f +/- %f' % (np.mean(f1), np.std(f1)))
    print('Cohens kappa: %f +/- %f' % (np.mean(kappa), np.std(kappa)))
    print('ROC AUC: %f +/- %f' % (np.mean(auc), np.std(auc)))


def run(data_dir, model, classifier, validation_dir, validation_split, n_splits, n_neighbors):
    if n_splits is None:
        run_one_shot(data_dir, model, classifier, validation_dir, validation_split, int(n_neighbors))
    else:
        run_cv(data_dir, model, classifier, validation_dir, validation_split, int(n_splits), int(n_neighbors))


def main():
    parser = argparse.ArgumentParser(
        description='Train xDNN with CNN Feature Extractor'
    )
    parser.add_argument('--data-dir', help='Training dataset path', required=True)
    parser.add_argument('--model', help='Select model to feature extractor', required=True)
    parser.add_argument('--classifier', help='Select the Classifier method', required=True)
    parser.add_argument('--validation-dir', help='Validation dataset path', default=None)
    parser.add_argument('--validation-split', help='Percentage dataset to validation', default=0.2)
    parser.add_argument('--n-splits', help='Number of splits to K-Fold Cross Validation', default=None)
    parser.add_argument('--n-neighbors', help='Number of neighbors to use by default for kneighbors queries when knn '
                                              'is selected as classifier', default=5)

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
