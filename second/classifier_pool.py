from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import numpy as np
import util

# TODO: Gaussian Process, Kernel Ridge Regression and Multilayer Perceptron can be implemented.

CREATE_SUBMISSION_FILE = True
N_JOBS = 8 # Num of CPU cores for parallel processing.
SUBMISSION_FILE_SUFFIX = '' # A one-word-descriptor for the experiment
PRINT_ESTIMATOR_RESULTS = False # If True, prints results for all possible configurations.

def svm(training_feature_matrix, training_targets, test_feature_matrix):
	from sklearn.svm import SVC
	# Parameter grid
	param_grid = {
		'kernel': ['rbf'],
		'gamma': np.logspace(-8, 1, 10),
		'C': np.logspace(-4, 6, 10),
	}
	svm = SVC(probability=True)
	clf = GridSearchCV(estimator=svm, param_grid=param_grid, n_jobs=N_JOBS, cv=10, scoring='neg_log_loss',)
	clf.fit(training_feature_matrix, training_targets)

	if PRINT_ESTIMATOR_RESULTS == True:
	    for params, mean_score, scores in clf.grid_scores_:
	        print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params))

	print("SVM [%0.3f] - The best parameters are %s" % (-clf.best_score_, clf.best_params_))
	predicted_labels = clf.predict_proba(test_feature_matrix)

	if CREATE_SUBMISSION_FILE == True:
		util.create_submission_file(predicted_labels[:,1], 'submission_svm%s.csv' % (SUBMISSION_FILE_SUFFIX))



def adaboost(training_feature_matrix, training_targets, test_feature_matrix):
	from sklearn.ensemble import AdaBoostClassifier
	from sklearn.tree import DecisionTreeClassifier
	# Parameter grid
	param_grid = {
		"learning_rate": [1e-4, 1e-3, 1e-2, 1e-1, 1, 2],
		'n_estimators': [25, 50, 100, 200],
	}
	adaboost = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3))
	clf = GridSearchCV(estimator=adaboost, param_grid=param_grid, n_jobs=N_JOBS, cv=10, scoring='neg_log_loss',)
	clf.fit(training_feature_matrix, training_targets)

	if PRINT_ESTIMATOR_RESULTS == True:
	    for params, mean_score, scores in clf.grid_scores_:
	        print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params))

	print("ADABOOST [%0.3f] - The best parameters are %s" % (-clf.best_score_, clf.best_params_))
	predicted_labels = clf.predict_proba(test_feature_matrix)

	if CREATE_SUBMISSION_FILE == True:
		util.create_submission_file(predicted_labels[:,1], 'submission_adaboost%s.csv' % (SUBMISSION_FILE_SUFFIX))



def knn(training_feature_matrix, training_targets, test_feature_matrix):
	from sklearn.neighbors import KNeighborsClassifier
	# Parameter grid
	param_grid = {
		"n_neighbors": [1,3,7,15,20],
	}
	knn = KNeighborsClassifier()
	clf = GridSearchCV(estimator=knn, param_grid=param_grid, n_jobs=N_JOBS, cv=10, scoring='neg_log_loss',)
	clf.fit(training_feature_matrix, training_targets)

	if PRINT_ESTIMATOR_RESULTS == True:
	    for params, mean_score, scores in clf.grid_scores_:
	        print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params))

	print("KNN [%0.3f] - The best parameters are %s" % (-clf.best_score_, clf.best_params_))
	predicted_labels = clf.predict_proba(test_feature_matrix)

	if CREATE_SUBMISSION_FILE == True:
		util.create_submission_file(predicted_labels[:,1], 'submission_knn%s.csv' % (SUBMISSION_FILE_SUFFIX))



def random_forest(training_feature_matrix, training_targets, test_feature_matrix):
	from sklearn.ensemble import RandomForestClassifier
	from scipy.stats import randint as sp_randint
	# Parameter distribution for random search.
	param_dist= {
		"max_depth": [2, 5, None],
		"max_features": sp_randint(1, min(300, training_feature_matrix.shape[1])),
		"min_samples_split": sp_randint(1, 20),
		"min_samples_leaf": sp_randint(1, 20),
		'n_estimators': [10, 50, 100, 200],
		'bootstrap': [True, False],
	}
	r_forest = RandomForestClassifier(random_state=1)
	n_iter_search = 500
	clf = RandomizedSearchCV(estimator=r_forest, param_distributions=param_dist, n_iter=n_iter_search, n_jobs=N_JOBS, cv=10, random_state=1, scoring='neg_log_loss')
	clf.fit(training_feature_matrix, training_targets)

	if PRINT_ESTIMATOR_RESULTS == True:
	    for params, mean_score, scores in clf.grid_scores_:
	        print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params))

	print("RANDOM FOREST [%0.3f] - The best parameters are %s" % (-clf.best_score_, clf.best_params_))
	predicted_labels = clf.predict_proba(test_feature_matrix)

	if CREATE_SUBMISSION_FILE == True:
		util.create_submission_file(predicted_labels[:,1], 'submission_random_forest%s.csv' % (SUBMISSION_FILE_SUFFIX))
