#!/usr/bin/env python -W ignore::DeprecationWarning
import numpy as np
import util
from sklearn import preprocessing, decomposition, manifold
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# A one-word-descriptor for the experiment 
SUBMISSION_FILE_SUFFIX = 'final'

# Data hyper-parameters
TRAIN_COUNT = 278
TEST_COUNT = 138

# Alredy preprocessed data. Data files must reside under DATA_PATH (see util.py)
# If it is False, the raw data is loaded and preprocessed from scratch.
USE_PREPROCESSED_DATA = False #('train_full_reduced.npy', 'test_full_reduced.npy')

# If USE_PREPROCESSED_DATA = False, then raw data of the provided z-planes 
# will be loaded and preprocessed.
#PLANES_Z = [20, 35, 56, 70, 87, 97, 107, 117, 127, 137]
PLANES_Z = list(range(15,175)) # Load all planes.

# Regression hyper-parameters
REDUCED_DIM = 50
MIX_TRAIN_TEST_FOR_REDUCTION = True
ESTIMATOR_POOL = {
    'svm_rbf': True,
    'huber': False,
    'lasso': False,
    'decision_tree': False,
    'random_forest': False,
    'adaboost': False,
    'gradientBR': False, # GradientBoostingRegressor
}
USE_FEATURE_POOL = False
# Specifies corresponding dimensionalites. Active if it is larger than 0.
# Calculates the sample coordinates in the lower-dimensional space for every
# technique. Simply concatenates them and creates a feature matrix for the classifiers.
FEATURE_POOL = {
    'tsne': 10,
    'lda': 10,
    'pca': 50,
    'mds': 0
}
PRINT_ESTIMATOR_RESULTS = False # If True, prints results of all possible configurations.
N_JOBS = 16 # Num of CPU cores for parallel processing.

def load_planes():
    training_planes = []
    for i in range(len(PLANES_Z)):
        z = PLANES_Z[i]
        print('Loading training plane %d/%d (z = %d)' % (i + 1, len(PLANES_Z), z))
        training_planes.append(util.nonzero_rv(util.load_all_z_planes(z)))

    test_planes = []
    for i in range(len(PLANES_Z)):
        z = PLANES_Z[i]
        print('Loading test plane %d/%d (z = %d)' % (i + 1, len(PLANES_Z), z))
        test_planes.append(util.nonzero_rv(util.load_all_z_planes(z, False)))

    return (np.concatenate(training_planes, axis=0), np.concatenate(test_planes, axis=0))

def main():
    #########
    # Prepare feature matrices.
    #########
    normalizer = preprocessing.StandardScaler()
    training_targets = util.load_refs() # Load targets (i.e. ages)
    # Load train & test data.
    if not USE_PREPROCESSED_DATA:
        print('Loading and preprocessing raw data.')
        training_planes, test_planes = load_planes()
        training_planes = training_planes.T
        test_planes = test_planes.T
        all_planes = np.vstack((training_planes, test_planes))
        if MIX_TRAIN_TEST_FOR_REDUCTION == True:
            print('Using both training and test data for reduction.')
            # Normalize data.
            all_planes = normalizer.fit_transform(all_planes)
            # Apply data reduction on both train and test data.
            # Note that this is not the best practice.
            all_planes = decomposition.PCA(n_components=REDUCED_DIM, whiten=False).fit_transform(all_planes)
            #all_planes = decomposition.KernelPCA(n_components=REDUCED_DIM, kernel='rbf').fit_transform(all_planes)
            training_planes = all_planes[:TRAIN_COUNT,:]
            test_planes = all_planes[TRAIN_COUNT:,:]
        else:
            print('Using only training data for reduction.')
            # Normalize data.
            training_planes = normalizer.fit_transform(training_planes)
            test_planes = normalizer.transform(test_planes)
            pca_decomp = decomposition.PCA(n_components=REDUCED_DIM, whiten=False)
            training_planes = pca_decomp.fit_transform(training_planes)
            test_planes = pca_decomp.transform(test_planes)
            all_planes = np.vstack((training_planes, test_planes))
    else:
        print('Using preprocessed-data.')
        training_planes, test_planes = util.load_preprocessed_data(USE_PREPROCESSED_DATA)
        training_planes = training_planes.T
        test_planes = test_planes.T
        # Normalize data.
        if MIX_TRAIN_TEST_FOR_REDUCTION == True:
            all_planes = np.vstack((training_planes, test_planes))
            all_planes = normalizer.fit_transform(all_planes)
            training_planes = all_planes[:TRAIN_COUNT,:]
            test_planes = all_planes[TRAIN_COUNT:,:]
        else:
            training_planes = normalizer.fit_transform(training_planes.T)
            test_planes = normalizer.transform(test_planes.T)
            all_planes = np.vstack((training_planes, test_planes))
    print('Data dimensionality ' + str(all_planes.shape))

    # Extract Features
    if USE_FEATURE_POOL == True:
        tsne_features = np.zeros((TRAIN_COUNT+TEST_COUNT, 0))
        lda_features = np.zeros((TRAIN_COUNT+TEST_COUNT, 0))
        pca_features = np.zeros((TRAIN_COUNT+TEST_COUNT, 0))
        mds_features = np.zeros((TRAIN_COUNT+TEST_COUNT, 0))
        if FEATURE_POOL['tsne'] > 0:
            print('Running t-SNE [%d]' % FEATURE_POOL['tsne'])
            tsne_features = manifold.TSNE(n_components=FEATURE_POOL['tsne'], init='pca', random_state=0).fit_transform(all_planes)
        if FEATURE_POOL['lda'] > 0:
            print('Running LDA [%d]' % FEATURE_POOL['lda'])
            lda = LinearDiscriminantAnalysis(n_components=FEATURE_POOL['lda'])
            lda_features_training = lda.fit_transform(training_planes, training_targets) #Supervised
            lda_features_test = lda.transform(test_planes)
            lda_features = np.vstack((lda_features_training, lda_features_test))
        if FEATURE_POOL['pca'] > 0:
            if REDUCED_DIM > FEATURE_POOL['pca']:
                print('Running PCA [%d]' % FEATURE_POOL['pca'])
                pca_features = decomposition.PCA(n_components=FEATURE_POOL['pca'], whiten=False).fit_transform(all_planes)
            else:
                print('Using old PCA [%d]' % FEATURE_POOL['pca'])
                pca_features = all_planes # Use already reduced data. 
        if FEATURE_POOL['mds'] > 0:
            print('Running MDS [%d]' % FEATURE_POOL['mds'])
            mds_features = manifold.MDS(n_components=FEATURE_POOL['mds'], n_init=1, max_iter=1000).fit_transform(all_planes)

        # Construct feature matrix by concatenation
        feature_matrix = np.hstack((tsne_features, lda_features, pca_features, mds_features))
        training_feature_matrix = all_planes[:TRAIN_COUNT,:]
        test_feature_matrix = all_planes[TRAIN_COUNT:,:]
    else:
        feature_matrix = all_planes
        training_feature_matrix = training_planes
        test_feature_matrix = test_planes

    print('Feature matrix dimensionality ' + str(feature_matrix.shape))
    #########
    # Grid Search and Regression
    #########
    # SVM (RBF Kernel)
    if ESTIMATOR_POOL['svm_rbf'] == True:
        print('Doing SVM (RBF) Regression...')
        from sklearn.svm import SVR
        # Parameter grid
        Cs = np.logspace(-4, 6, 20)
        gammas = np.logspace(-8, 1, 20)
        svr_rbf = SVR(kernel='rbf')
        clf = GridSearchCV(estimator=svr_rbf, param_grid=dict(C=Cs, gamma=gammas), n_jobs=N_JOBS, cv=10, scoring='neg_mean_squared_error')
        clf.fit(training_feature_matrix, training_targets)
        if PRINT_ESTIMATOR_RESULTS == True:
            for params, mean_score, scores in clf.grid_scores_:
                print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params))
        print("SVM(RBF) [%0.2f] - The best parameters are %s" % (-clf.best_score_, clf.best_params_))
        predicted_labels = clf.predict(test_feature_matrix)
        util.create_submission_file(predicted_labels, 'submission_rbf_svm_%s.csv' % (SUBMISSION_FILE_SUFFIX))
    # Huber Regression
    if ESTIMATOR_POOL['huber'] == True:
        print('Doing Huber Regression...')
        from sklearn.linear_model import HuberRegressor
        alphas = np.linspace(0.1, 5, 10)
        epsilons = np.linspace(1, 10, 10)
        huber = HuberRegressor(fit_intercept=True, max_iter=500)
        clf = GridSearchCV(estimator=huber, param_grid=dict(alpha=alphas, epsilon=epsilons), n_jobs=N_JOBS, cv=10, scoring='neg_mean_squared_error')
        clf.fit(training_feature_matrix, training_targets)
        if PRINT_ESTIMATOR_RESULTS == True:
            for params, mean_score, scores in clf.grid_scores_:
                print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params))
        print("Huber Regression [%0.2f] - The best parameters are %s" % (-clf.best_score_, clf.best_params_))
        predicted_labels = clf.predict(test_feature_matrix)
        util.create_submission_file(predicted_labels, 'submission_huber_%s.csv' % (SUBMISSION_FILE_SUFFIX))
    # LASSO
    if ESTIMATOR_POOL['lasso'] == True:
        print('Doing LASSO Regression...')
        from sklearn.linear_model import Lasso
        alphas = np.linspace(0.1, 6, 20)
        lasso = Lasso(alpha=0.1, copy_X=True, fit_intercept=False, max_iter=1000)
        clf = GridSearchCV(estimator=lasso, param_grid=dict(alpha=alphas), n_jobs=N_JOBS, cv=10, scoring='neg_mean_squared_error')
        clf.fit(training_feature_matrix, training_targets)
        if PRINT_ESTIMATOR_RESULTS == True:
            for params, mean_score, scores in clf.grid_scores_:
                print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params))
        print("LASSO Regression [%0.2f] - The best parameters are %s" % (-clf.best_score_, clf.best_params_))
        predicted_labels = clf.predict(test_feature_matrix)
        util.create_submission_file(predicted_labels, 'submission_lasso_%s.csv' % (SUBMISSION_FILE_SUFFIX))
    # Decision Tree
    if ESTIMATOR_POOL['decision_tree'] == True:
        print('Doing Decision Tree Regression...')
        # Use random search instead of grid search.
        from scipy.stats import randint as sp_randint
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.grid_search import RandomizedSearchCV
        decision_tree = DecisionTreeRegressor(random_state=0)
        param_dist = {"max_depth": [3, 5, None],
              "max_features": sp_randint(1, 20),
              "min_samples_split": sp_randint(1, 20),
              "min_samples_leaf": sp_randint(1, 20)
              }
        n_iter_search = 500
        clf = RandomizedSearchCV(decision_tree, param_distributions=param_dist, n_iter=n_iter_search, n_jobs=N_JOBS, cv=10, scoring='neg_mean_squared_error')
        #clf = GridSearchCV(estimator=decision_tree, param_grid=dict(), n_jobs=-1, cv=5, scoring='neg_mean_squared_error')
        clf.fit(training_feature_matrix, training_targets)
        if PRINT_ESTIMATOR_RESULTS == True:
            for params, mean_score, scores in clf.grid_scores_:
                print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params))
        print("Decision Tree Regression [%0.2f] - The best parameters are %s" % (-clf.best_score_, clf.best_params_))
        predicted_labels = clf.predict(test_feature_matrix)
        util.create_submission_file(predicted_labels, 'submission_decision_tree_%s.csv' % (SUBMISSION_FILE_SUFFIX))
    # Random Forest
    if ESTIMATOR_POOL['random_forest'] == True:
        print('Doing Random Forest Regression...')
        # Use random search instead of grid search.
        from scipy.stats import randint as sp_randint
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.grid_search import RandomizedSearchCV
        forest = RandomForestRegressor(random_state=0)
        param_dist = {"max_depth": [3, 5, None],
              "max_features": sp_randint(1, 20),
              "min_samples_split": sp_randint(1, 20),
              "min_samples_leaf": sp_randint(1, 20),
              'n_estimators': [10, 25, 50, 100],
              'bootstrap': [True, False],
              }
        n_iter_search = 200
        clf = RandomizedSearchCV(forest, param_distributions=param_dist, n_iter=n_iter_search, n_jobs=N_JOBS, cv=10, scoring='neg_mean_squared_error')
        #clf = GridSearchCV(estimator=decision_tree, param_grid=dict(), n_jobs=-1, cv=5, scoring='neg_mean_squared_error')
        clf.fit(training_feature_matrix, training_targets)
        if PRINT_ESTIMATOR_RESULTS == True:
            for params, mean_score, scores in clf.grid_scores_:
                print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params))
        print("Random Forest Regression [%0.2f] - The best parameters are %s" % (-clf.best_score_, clf.best_params_))
        predicted_labels = clf.predict(test_feature_matrix)
        util.create_submission_file(predicted_labels, 'submission_random_forest_%s.csv' % (SUBMISSION_FILE_SUFFIX))
    # AdaBoost
    if ESTIMATOR_POOL['adaboost'] == True:
        print('Doing AdaBoost Regression...')
        from sklearn.ensemble import AdaBoostRegressor
        adaboost = AdaBoostRegressor(loss='square', random_state=0)
        param_dist = {
              "learning_rate": [1e-4, 1e-3, 1e-2, 1e-1, 1, 2],
              'n_estimators': [10, 25, 50, 100, 200],
              }
        clf = GridSearchCV(estimator=adaboost, param_grid=param_dist, n_jobs=N_JOBS, cv=10, scoring='neg_mean_squared_error')
        clf.fit(training_feature_matrix, training_targets)
        if PRINT_ESTIMATOR_RESULTS == True:
            for params, mean_score, scores in clf.grid_scores_:
                print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params))
        print("AdaBoost Regression [%0.2f] - The best parameters are %s" % (-clf.best_score_, clf.best_params_))
        predicted_labels = clf.predict(test_feature_matrix)
        util.create_submission_file(predicted_labels, 'submission_adaboost_%s.csv' % (SUBMISSION_FILE_SUFFIX))
    # GradientBoostingRegressor
    if ESTIMATOR_POOL['gradientBR'] == True:
        from sklearn.ensemble import GradientBoostingRegressor
        param_dist = {
            'loss' : ['ls', 'lad', 'huber'],
            'n_estimators': [20, 100, 200],
            'learning_rate': [1e-3, 1e-1, 1],
            'max_depth': [3, 6],
            'min_samples_split': [2, 5],
            }
        gbr = GradientBoostingRegressor(random_state=0)
        clf = GridSearchCV(estimator=gbr, param_grid=param_dist, n_jobs=N_JOBS, cv=6, scoring='neg_mean_squared_error')
        clf.fit(training_feature_matrix, training_targets)
        if PRINT_ESTIMATOR_RESULTS == True:
            for params, mean_score, scores in clf.grid_scores_:
                print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params))
        print("GradientBoostingRegressor Regression [%0.2f] - The best parameters are %s" % (-clf.best_score_, clf.best_params_))
        predicted_labels = clf.predict(test_feature_matrix)
        util.create_submission_file(predicted_labels, 'submission_gbr_%s.csv' % (SUBMISSION_FILE_SUFFIX))
        
if __name__ == '__main__':
    main()