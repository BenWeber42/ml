#!/usr/bin/env python -W ignore::DeprecationWarning
import numpy as np
import util
from sklearn import preprocessing
import classifier_pool as classifiers
from sklearn import preprocessing, decomposition
from sklearn.pipeline import Pipeline
from pipeline import (
    DataPipeline,
    FeatureExtractors
)

# Data hyper-parameters
TRAIN_COUNT = 278
TEST_COUNT = 138

REDUCED_DIM = 30

#PLANES_Z = [20, 35, 56, 70, 87, 97, 107, 117, 127, 137]
PLANES_Z = list(range(20,170)) # Load all planes.

ESTIMATOR_POOL = {
    'svm': True,
    'random_forest': True,
    'adaboost': True,
    'knn': True,
}

def load_z_planes():
    training_planes = []
    for i in range(len(PLANES_Z)):
        z = PLANES_Z[i]
        print('Loading training plane %d/%d (z = %d)' % (i + 1, len(PLANES_Z), z))
        training_planes.append(util.load_all_z_planes(z))

    test_planes = []
    for i in range(len(PLANES_Z)):
        z = PLANES_Z[i]
        print('Loading test plane %d/%d (z = %d)' % (i + 1, len(PLANES_Z), z))
        test_planes.append(util.load_all_z_planes(z, False))

    return (np.concatenate(training_planes, axis=0), np.concatenate(test_planes, axis=0))

def main():
    ####################################
    # Prepare feature matrices.
    ####################################
    normalizer = preprocessing.StandardScaler()
    training_targets = util.load_refs() # Load targets

    data_pipeline = DataPipeline()
    feature_extractors = FeatureExtractors(data_pipeline, {
        FeatureExtractors.HISTOGRAMS: {
            'partitions': [(9, 9, 9)],
            'bins': [15],
            'interval': [(1, 2000)]
        },
        #FeatureExtractors.PCA: {}
    })
    pipeline = Pipeline([
        ('data_dict_builder', data_pipeline.data_dict_builder),
        ('feature_extraction', feature_extractors.get_pipeline()),
    ])
    hist_features_all = pipeline.transform(data_pipeline.all_data)
    hist_features_all = decomposition.PCA(n_components=REDUCED_DIM, whiten=False).fit_transform(hist_features_all)

    pca_features_all = decomposition.PCA(n_components=REDUCED_DIM, whiten=False).fit_transform(data_pipeline.all_data)

    features_all = np.hstack((hist_features_all, pca_features_all))
    features_all = normalizer.fit_transform(features_all)
    training_feature_matrix = features_all[:TRAIN_COUNT,:]
    test_feature_matrix = features_all[TRAIN_COUNT:,:]

    '''
    # Load train & test data.
    print('Loading and preprocessing raw data.')
    training_planes, test_planes = load_z_planes()
    training_planes = training_planes.T
    test_planes = test_planes.T
    all_planes = np.vstack((training_planes, test_planes))
    print('Data dimensionality ' + str(all_planes.shape))

    # TODO: Feature pool will come here
    print('Using both training and test data for reduction.')
    # Apply data reduction on both train and test data.
    all_planes = normalizer.fit_transform(all_planes)
    #all_planes = decomposition.KernelPCA(n_components=REDUCED_DIM, kernel='rbf').fit_transform(all_planes)
    all_planes = decomposition.PCA(n_components=REDUCED_DIM, whiten=False).fit_transform(all_planes)
    #all_planes = normalizer.fit_transform(all_planes)
    training_planes = all_planes[:TRAIN_COUNT,:]
    test_planes = all_planes[TRAIN_COUNT:,:]

    training_feature_matrix = training_planes
    test_feature_matrix = test_planes
    '''
    print('Training feature matrix dimensionality ' + str(training_feature_matrix.shape))
    print('Classification')
    if ESTIMATOR_POOL['svm'] == True:
        classifiers.svm(training_feature_matrix, training_targets, test_feature_matrix)
    if ESTIMATOR_POOL['random_forest'] == True:
        classifiers.random_forest(training_feature_matrix, training_targets, test_feature_matrix)
    if ESTIMATOR_POOL['knn'] == True:
        classifiers.knn(training_feature_matrix, training_targets, test_feature_matrix)
    if ESTIMATOR_POOL['adaboost'] == True:
        classifiers.adaboost(training_feature_matrix, training_targets, test_feature_matrix)

if __name__ == '__main__':
    main()
