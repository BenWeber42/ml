from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import hamming_loss, make_scorer
from os import environ as sysenv
import numpy as np
import util

# TODO: Gaussian Process, Kernel Ridge Regression and Multilayer Perceptron
# can be implemented.

CREATE_SUBMISSION_FILE = True
# Num of CPU cores for parallel processing.
N_JOBS = 8
if 'LBD_N_JOBS' in sysenv.keys():
    N_JOBS = int(sysenv['LBD_N_JOBS'])
    print('Using %d jobs' % N_JOBS)
# A one-word-descriptor for the experiment
SUBMISSION_FILE_SUFFIX = ''
# If True, prints results for all possible configurations.
PRINT_ESTIMATOR_RESULTS = False
# How many cross validation folds to do
CV_N = 20
# Scoring
SCORING = make_scorer(hamming_loss, greater_is_better=False)


def svm(training_feature_matrix, training_targets, test_feature_matrix):
    from sklearn.svm import SVC
    # Parameter grid
    param_grid = {
        'kernel': ['rbf'],
        'gamma': np.logspace(-8, 1, 7),
        'C': np.logspace(-4, 6, 7),
    }
    svm = SVC(probability=True)
    clf = GridSearchCV(
        estimator=svm,
        param_grid=param_grid,
        n_jobs=N_JOBS,
        pre_dispatch=N_JOBS,
        cv=CV_N,
        scoring=SCORING
    )
    clf.fit(training_feature_matrix, training_targets)

    if PRINT_ESTIMATOR_RESULTS is True:
        for (mean,std) in zip(clf.cv_results_['mean_test_score'], clf.cv_results_['std_test_score']):
            print("%0.3f (+/-%0.3f)" % (-mean,std))

    print("%s SVM: %0.3f (+/-%0.3f) Parameters %s"
          % (SUBMISSION_FILE_SUFFIX, -clf.best_score_, 
            clf.cv_results_['std_test_score'][clf.best_index_],
            clf.best_params_))

    predicted_labels = clf.predict(test_feature_matrix)
    if CREATE_SUBMISSION_FILE is True:
        util.create_submission_file(
            predicted_labels,
            'submission_svm%s.csv' % (SUBMISSION_FILE_SUFFIX)
        )


def adaboost(training_feature_matrix, training_targets, test_feature_matrix):
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    # Parameter grid
    param_grid = {
        "learning_rate": [1e-4, 1e-3, 1e-2, 1e-1, 1],
        'n_estimators': [25, 50, 100],
    }
    adaboost = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3))
    clf = GridSearchCV(
        estimator=adaboost,
        param_grid=param_grid,
        n_jobs=N_JOBS,
        pre_dispatch=N_JOBS,
        cv=CV_N,
        scoring=SCORING
    )
    clf.fit(training_feature_matrix, training_targets)

    if PRINT_ESTIMATOR_RESULTS is True:
        for (mean,std) in zip(clf.cv_results_['mean_test_score'], clf.cv_results_['std_test_score']):
            print("%0.3f (+/-%0.3f)" % (-mean,std))

    print("%s ADABOOST: %0.3f (+/-%0.3f) Parameters %s"
          % (SUBMISSION_FILE_SUFFIX, -clf.best_score_, 
            clf.cv_results_['std_test_score'][clf.best_index_],
            clf.best_params_))

    predicted_labels = clf.predict(test_feature_matrix)
    if CREATE_SUBMISSION_FILE is True:
        util.create_submission_file(
            predicted_labels,
            'submission_adaboost%s.csv' % (SUBMISSION_FILE_SUFFIX)
        )


def knn(training_feature_matrix, training_targets, test_feature_matrix):
    from sklearn.neighbors import KNeighborsClassifier
    # Parameter grid
    param_grid = {
        "n_neighbors": [1, 3, 7, 15, 20]
    }
    knn = KNeighborsClassifier()
    clf = GridSearchCV(
        estimator=knn,
        param_grid=param_grid,
        n_jobs=N_JOBS,
        pre_dispatch=N_JOBS,
        cv=CV_N,
        scoring=SCORING
    )
    clf.fit(training_feature_matrix, training_targets)

    if PRINT_ESTIMATOR_RESULTS is True:
        for (mean,std) in zip(clf.cv_results_['mean_test_score'], clf.cv_results_['std_test_score']):
            print("%0.3f (+/-%0.3f)" % (-mean,std))

    print("%s KNN: %0.3f (+/-%0.3f) Parameters %s"
          % (SUBMISSION_FILE_SUFFIX, -clf.best_score_, 
            clf.cv_results_['std_test_score'][clf.best_index_],
            clf.best_params_))

    predicted_labels = clf.predict(test_feature_matrix)
    if CREATE_SUBMISSION_FILE is True:
        util.create_submission_file(
            predicted_labels,
            'submission_knn%s.csv' % (SUBMISSION_FILE_SUFFIX)
        )
    return clf


def random_forest(
    training_feature_matrix,
    training_targets,
    test_feature_matrix
):
    from sklearn.ensemble import RandomForestClassifier
    from scipy.stats import randint as sp_randint
    # Parameter distribution for random search.
    param_dist = {
        "max_depth": [2, 5, None],
        "max_features": sp_randint(
            1,
            min(300, training_feature_matrix.shape[1])
        ),
        "min_samples_split": sp_randint(1, 20),
        "min_samples_leaf": sp_randint(1, 20),
        'n_estimators': [10, 50, 100, 200, 300],
        'bootstrap': [True, False],
    }
    r_forest = RandomForestClassifier(random_state=1)
    n_iter_search = 200
    clf = RandomizedSearchCV(
        estimator=r_forest,
        param_distributions=param_dist,
        n_iter=n_iter_search,
        n_jobs=N_JOBS,
        pre_dispatch=N_JOBS,
        cv=CV_N,
        random_state=1,
        scoring=SCORING
    )
    clf.fit(training_feature_matrix, training_targets)

    if PRINT_ESTIMATOR_RESULTS is True:
        for (mean,std) in zip(clf.cv_results_['mean_test_score'], clf.cv_results_['std_test_score']):
            print("%0.3f (+/-%0.3f)" % (-mean,std))

    print("%s RANDOM FOREST: %0.3f (+/-%0.3f) Parameters %s"
          % (SUBMISSION_FILE_SUFFIX, -clf.best_score_, 
            clf.cv_results_['std_test_score'][clf.best_index_],
            clf.best_params_))

    predicted_labels = clf.predict(test_feature_matrix)
    if CREATE_SUBMISSION_FILE is True:
        util.create_submission_file(
            predicted_labels,
            'submission_random_forest%s.csv' % (SUBMISSION_FILE_SUFFIX)
        )
    return clf

def extra_trees_classifier(
    training_feature_matrix,
    training_targets,
    test_feature_matrix
):
    from sklearn.ensemble import ExtraTreesClassifier
    from scipy.stats import randint as sp_randint
    # Parameter distribution for random search.
    param_dist = {
        "criterion": ["gini", "entropy"],
        "max_features": sp_randint(
            1,
            min(300, training_feature_matrix.shape[1])
        ),
        "max_depth": [2, 5, None],
        "min_samples_split": sp_randint(1, 20),
        "min_samples_leaf": sp_randint(1, 20),
        'n_estimators': [10, 50, 100, 200, 300],
    }
    etc = ExtraTreesClassifier(random_state=1)
    n_iter_search = 200
    clf = RandomizedSearchCV(
        estimator=etc,
        param_distributions=param_dist,
        n_iter=n_iter_search,
        n_jobs=N_JOBS,
        pre_dispatch=N_JOBS,
        cv=CV_N,
        random_state=1,
        scoring=SCORING
    )
    clf.fit(training_feature_matrix, training_targets)

    if PRINT_ESTIMATOR_RESULTS is True:
        for (mean,std) in zip(clf.cv_results_['mean_test_score'], clf.cv_results_['std_test_score']):
            print("%0.3f (+/-%0.3f)" % (-mean,std))

    print("%s EXTRA_TREES_CLASSIFIER: %0.3f (+/-%0.3f) Parameters %s"
          % (SUBMISSION_FILE_SUFFIX, -clf.best_score_, 
            clf.cv_results_['std_test_score'][clf.best_index_],
            clf.best_params_))

    predicted_labels = clf.predict(test_feature_matrix)
    if CREATE_SUBMISSION_FILE is True:
        util.create_submission_file(
            predicted_labels,
            'extra_trees_classifier%s.csv' % (SUBMISSION_FILE_SUFFIX)
        )