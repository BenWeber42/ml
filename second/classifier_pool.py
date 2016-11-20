from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# TODO: Gaussian Process, Kernel Ridge Regression and Multilayer Perceptron can be implemented.

def run_svm(training_feature_matrix, training_targets, test_feature_matrix):
	from sklearn.svm import SVR
	# Parameter grid
	param_grid = {
		'kernel' = ['rbf', 'linear'],
		'degree' = [2,3,5],
		'gamma' = np.logspace(-8, 1, 15),
		'C' = np.logspace(-4, 6, 15),
	}
	svm = SVC()
	clf = GridSearchCV(estimator=svm, param_grid=param_grid, n_jobs=N_JOBS, cv=10)
	clf.fit(training_feature_matrix, training_targets)

	if PRINT_ESTIMATOR_RESULTS == True:
	    for params, mean_score, scores in clf.grid_scores_:
	        print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params))

	print("SVM [%0.2f] - The best parameters are %s" % (-clf.best_score_, clf.best_params_))
	predicted_labels = clf.predict(test_feature_matrix)
	
	if CREATE_SUBMISSION_FILE == True:
		util.create_submission_file(predicted_labels, 'submission_svm_%s.csv' % (SUBMISSION_FILE_SUFFIX))



def run_adaboost(training_feature_matrix, training_targets, test_feature_matrix):
	from sklearn.ensemble.AdaBoostClassifier
	# Parameter grid
	param_grid = {
		"learning_rate": [1e-4, 1e-3, 1e-2, 1e-1, 1, 2],
		'n_estimators': [10, 25, 50, 100, 200],
	}
	adaboost = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3))
	clf = GridSearchCV(estimator=adaboost, param_grid=param_grid, n_jobs=N_JOBS, cv=10)
	clf.fit(training_feature_matrix, training_targets)

	if PRINT_ESTIMATOR_RESULTS == True:
	    for params, mean_score, scores in clf.grid_scores_:
	        print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params))

	print("Adaboost [%0.2f] - The best parameters are %s" % (-clf.best_score_, clf.best_params_))
	predicted_labels = clf.predict(test_feature_matrix)
	
	if CREATE_SUBMISSION_FILE == True:
		util.create_submission_file(predicted_labels, 'submission_adaboost_%s.csv' % (SUBMISSION_FILE_SUFFIX))



def run_knn(training_feature_matrix, training_targets, test_feature_matrix):
	from sklearn.neighbors.KNeighborsClassifier
	# Parameter grid
	param_grid = {
		"n_neighbors ": [1,3,5,9,15],
	}
	knn = KNeighborsClassifier()
	clf = GridSearchCV(estimator=knn, param_grid=param_grid, n_jobs=N_JOBS, cv=10)
	clf.fit(training_feature_matrix, training_targets)

	if PRINT_ESTIMATOR_RESULTS == True:
	    for params, mean_score, scores in clf.grid_scores_:
	        print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params))

	print("KNN [%0.2f] - The best parameters are %s" % (-clf.best_score_, clf.best_params_))
	predicted_labels = clf.predict(test_feature_matrix)
	
	if CREATE_SUBMISSION_FILE == True:
		util.create_submission_file(predicted_labels, 'submission_knn_%s.csv' % (SUBMISSION_FILE_SUFFIX))
	


def run_random_forest(training_feature_matrix, training_targets, test_feature_matrix):
	from sklearn.ensemble.RandomForestClassifier
	# Parameter distribution for random search.
	param_dist= {
		"max_depth": [2, 5, None],
		"max_features": sp_randint(1, 20),
		"min_samples_split": sp_randint(1, 20),
		"min_samples_leaf": sp_randint(1, 20),
		'n_estimators': [10, 50, 100, 200],
		'bootstrap': [True, False],
	}
	r_forest = RandomForestClassifier()
	n_iter_search = 200
	clf = RandomizedSearchCV(estimator=r_forest, param_distributions=param_dist, n_iter=n_iter_search, n_jobs=N_JOBS, cv=10)
	clf.fit(training_feature_matrix, training_targets)

	if PRINT_ESTIMATOR_RESULTS == True:
	    for params, mean_score, scores in clf.grid_scores_:
	        print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params))

	print("Random Forest [%0.2f] - The best parameters are %s" % (-clf.best_score_, clf.best_params_))
	predicted_labels = clf.predict(test_feature_matrix)
	
	if CREATE_SUBMISSION_FILE == True:
		util.create_submission_file(predicted_labels, 'submission_random_forest_%s.csv' % (SUBMISSION_FILE_SUFFIX))