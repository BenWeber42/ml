#!/usr/bin/env python -W ignore::DeprecationWarning
import numpy as np
import util
import classifier_pool as classifiers
from sklearn.pipeline import Pipeline
from pipeline import (
    DataPipeline,
    FeatureExtractors
)
np.random.seed(1)
# Data hyper-parameters
TRAIN_COUNT = 278
TEST_COUNT = 138

PLANES_Z = list(range(20, 170))  # Load planes.

ESTIMATOR_POOL = {
    'svm': False,
    'random_forest': True,
    'adaboost': False,
    'knn': False,
}


def main():
    ####################################
    # Prepare feature matrices.
    ####################################
    print('Loading Data.')
    training_targets = util.load_refs()  # Load targets

    data_pipeline = DataPipeline()
    feature_extractors = FeatureExtractors(data_pipeline, {
        FeatureExtractors.HISTOGRAMS: {
            'partitions': [(9, 9, 9)],
            'bins': [30],
            'interval': [(1, 2000)]
        },
    })
    pipeline = Pipeline([
        ('data_dict_builder', data_pipeline.data_dict_builder),
        ('feature_extraction', feature_extractors.get_pipeline()),
    ])
    print('Creating Feature Matrix.')
    features_all = pipeline.transform(data_pipeline.all_data)

    training_feature_matrix = features_all[:TRAIN_COUNT, :]
    test_feature_matrix = features_all[TRAIN_COUNT:, :]

    print('Training feature matrix dimensionality '
          + str(training_feature_matrix.shape))
    print('Classification.')
    if ESTIMATOR_POOL['svm'] is True:
        classifiers.svm(
            training_feature_matrix, training_targets, test_feature_matrix)
    if ESTIMATOR_POOL['random_forest'] is True:
        classifiers.random_forest(
            training_feature_matrix, training_targets, test_feature_matrix)
    if ESTIMATOR_POOL['knn'] is True:
        classifiers.knn(
            training_feature_matrix, training_targets, test_feature_matrix)
    if ESTIMATOR_POOL['adaboost'] is True:
        classifiers.adaboost(
            training_feature_matrix, training_targets, test_feature_matrix)


if __name__ == '__main__':
    main()
