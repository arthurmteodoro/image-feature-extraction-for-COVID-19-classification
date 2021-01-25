from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
import argparse
import os
import sys
import feature_extractor
from extract_feature import extract_feature
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV
import pandas as pd

params_search = {'svm-linear': {'kernel': ['linear'], 'C': [1, 10, 100, 1000], 'class_weight': ['balanced', None]},
                 'svm': {'kernel': ['rbf'], 'C': [1, 10, 100, 1000], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                         'class_weight': ['balanced', None]},
                 'knn': {'n_neighbors': [1, 3, 5, 10, 15], 'weights': ['uniform', 'distance'],
                         'leaf_size': [10, 20, 30, 40, 50], 'p': [1, 2, 3]}
                 }

base_estimator = {
    'svm': SVC(),
    'svm-linear': SVC(),
    'knn': KNeighborsClassifier()
}

classifiers_available = [
    'knn',
    'svm',
    'svm-linear'
]


def extract_features(data_dir, model):
    model, process_img = feature_extractor.get_feature_extractor(model)
    # model.summary()

    print('Extract Features from dataset')
    input_train = extract_feature(data_dir, model, process_img, verbose=1, validation_split=None)

    return input_train['Features'], input_train['Labels']


def run(data_dir, model, classifier, metrics):
    clf = base_estimator[classifier]
    x, y = extract_features(data_dir, model)

    grid_search = GridSearchCV(clf, params_search[classifier], scoring=metrics, refit=metrics[0],
                               n_jobs=-1, verbose=1)
    grid_search.fit(x, y)
    print('Best params')
    print()
    print(grid_search.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    print(grid_search.cv_results_.keys())
    means = grid_search.cv_results_['mean_test_accuracy']
    stds = grid_search.cv_results_['std_test_accuracy']
    for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Train xDNN with CNN Feature Extractor'
    )
    parser.add_argument('--data-dir', help='Training dataset path', required=True)
    parser.add_argument('--model', help='Select model to feature extractor', required=True)
    parser.add_argument('--classifier', help='Select the Classifier method', required=True)
    parser.add_argument('--metrics', help='Metrics to evaluate', nargs='*', default='accuracy')

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
