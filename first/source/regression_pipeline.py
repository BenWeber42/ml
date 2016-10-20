#!/usr/bin/env python
import numpy as np
import util
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, cross_val_score

# Data hyper-parameters
TRAIN_COUNT = 278
TEST_COUNT = 138
PLANES_Z = [20, 35, 56, 70, 87, 97, 107, 117, 127, 137]
# Regression hyper-parameters
REDUCED_DIM = 10
MIX_TRAIN_TEST_FOR_REDUCTION = False
ESTIMATORS = {
    'svm_rbf': False,
    'huber': True,
    'lasso': False,
}

def load_10_planes():
    train_planes = []
    for i in range(len(PLANES_Z)):
        z = PLANES_Z[i]
        print('Loading training plane %d/%d (z = %d)' % (i + 1, len(PLANES_Z), z))
        train_planes.append(util.nonzero_rv(util.load_all_z_planes(z)))

    test_planes = []
    for i in range(len(PLANES_Z)):
        z = PLANES_Z[i]
        print('Loading test plane %d/%d (z = %d)' % (i + 1, len(PLANES_Z), z))
        test_planes.append(util.nonzero_rv(util.load_all_z_planes(z, False)))

    return (np.concatenate(train_planes, axis=0), np.concatenate(test_planes, axis=0))

def main():
    #########
    # Prepare feature matrices.
    #########
    # Load train & test data.
    train_planes, test_planes = load_10_planes()
    normalizer = preprocessing.StandardScaler()
    training_targets = util.load_refs()
    if MIX_TRAIN_TEST_FOR_REDUCTION == True:
        print('Using both training and test data for reduction.')
        # Apply data reduction on both train and test data.
        # Note that this is not the best practice.
        planes = np.hstack((train_planes, test_planes))
        pc, _ = util.dense_pca(planes)
        reduced_planes = np.dot(planes.T, pc[:, :REDUCED_DIM])
        # Normalize data.
        reduced_planes = normalizer.fit_transform(reduced_planes)
        # Set training and test feature matrices which will be used
        # by the estimators.
        training_feature_matrix = reduced_planes[:TRAIN_COUNT,:]
        test_feature_matrix = reduced_planes[TRAIN_COUNT:,:]
    else:
        print('Using only training data for reduction.')
        pc, _ = util.dense_pca(train_planes)
        reduced_train_planes = np.dot(train_planes.T, pc[:, :REDUCED_DIM])
        reduced_test_planes = np.dot(test_planes.T, pc[:, :REDUCED_DIM])
        # Normalize data.
        training_feature_matrix = normalizer.fit_transform(reduced_train_planes)
        test_feature_matrix = normalizer.transform(reduced_test_planes)
    #########
    # Grid Search and Regression
    #########
    # SVM (RBF Kernel)
    if ESTIMATORS['svm_rbf'] == True:
        print('Doing SVM (RBF) Regression...')
        from sklearn.svm import SVR
        # Parameter grid
        Cs = np.logspace(-4, 6, 15)
        gammas = np.logspace(-8, 1, 15)
        svr_rbf = SVR(kernel='rbf')
        clf = GridSearchCV(estimator=svr_rbf, param_grid=dict(C=Cs, gamma=gammas), n_jobs=-1, cv=5, scoring='neg_mean_squared_error')
        clf.fit(training_feature_matrix, training_targets)
        for params, mean_score, scores in clf.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params))
        print("SVM(RBF) - The best parameters are %s with a score of %0.2f" % (clf.best_params_, clf.best_score_))
        predicted_labels = clf.predict(test_feature_matrix)
        util.create_submission_file(predicted_labels, 'submission_rbf_svm.csv')
    # Huber Regression
    if ESTIMATORS['huber'] == True:
        print('Doing Huber Regression...')
        from sklearn.linear_model import HuberRegressor
        alphas = np.linspace(0.1, 5, 10)
        epsilons = np.linspace(1, 10, 10)
        huber = HuberRegressor(fit_intercept=True, max_iter=500)
        clf = GridSearchCV(estimator=huber, param_grid=dict(alpha=alphas, epsilon=epsilons), n_jobs=-1, cv=5, scoring='neg_mean_squared_error')
        clf.fit(training_feature_matrix, training_targets)
        for params, mean_score, scores in clf.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params))
        print("Huber Regression - The best parameters are %s with a score of %0.2f" % (clf.best_params_, clf.best_score_))
        predicted_labels = clf.predict(test_feature_matrix)
        util.create_submission_file(predicted_labels, 'submission_huber.csv')

if __name__ == '__main__':
    main()