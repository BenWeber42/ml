#!/usr/bin/env python
import numpy as np
import util
from histograms import partitioned_histograms
from canny_edges import partitioned_canny_edges
import classifier_pool as classifiers
np.random.seed(1)

ESTIMATOR_POOL = {
    'random_forest': True,
    'knn': True,
    'extra_trees_classifier': True,
    'svm': False, # Does not support multi-label classification
    'adaboost': False, # Does not support multi-label classification
}

FEATURE_POOL = {
    'pca': True,
    'histogram': True,
    'canny_edges': True,
}

def main():
    ####################################
    # Prepare feature matrices.
    ####################################
    # If the corresponding feature extraction method is not used, then
    # this will be used as a replacement. 
    emptyDataset = np.empty(shape=(util.TRAIN_COUNT+util.TEST_COUNT, 0))

    print('Loading Targets.')
    training_targets = util.load_refs()  # Load targets

    if FEATURE_POOL['pca'] == True:
        print('Loading PCA.')
        pca_dataset = util.load_pca_dataset()
        pca_cfg = str(pca_dataset.shape[1])
    else:
        pca_dataset = emptyDataset
        pca_cfg = 'None'

    if FEATURE_POOL['histogram'] == True or FEATURE_POOL['canny_edges'] == True:
        print('Loading Data.')
        dataset = util.load_full_dataset()

    if FEATURE_POOL['histogram'] == True:
        print('Precomputing histograms.')
        histograms = [
            (
                partitioned_histograms(
                    dataset,
                    partitions=partitions,
                    interval=interval,
                    bins=bins
                ),
                {
                    'partitions': partitions,
                    'interval': interval,
                    'bins': bins
                }
            )
            for partitions in [
                (3, 3, 3),
                (6, 6, 6),
                (9, 9, 9)
            ]
            for interval in [
                (1, 2000)
            ]
            for bins in [
                15, 30, 35
            ]
        ]
    else:
        histograms = [[emptyDataset, 'None']]

    if FEATURE_POOL['canny_edges'] == True:
        print('Precomputing canny edges.')
        canny_edges = partitioned_canny_edges(dataset)
        canny_edges_cfg = 'default'
    else:
        canny_edges = emptyDataset
        canny_edges_cfg = 'None'
    
    # To better keep track of the submission files and output logs.
    featureConfigurationID = 0 
    for hist_data, hist_cfg in histograms:
        featureConfigurationID = featureConfigurationID + 1
        classifiers.SUBMISSION_FILE_SUFFIX = '(' + str(featureConfigurationID) + ')'

        feature_cfg = {
            'pca': pca_cfg,
            'histograms': hist_cfg,
            'canny_edges': canny_edges_cfg
        }

        print('Using features settings:')
        print(classifiers.SUBMISSION_FILE_SUFFIX + ' ' + str(feature_cfg))

        training_feature_matrix = np.hstack((
            hist_data[:util.TRAIN_COUNT],
            pca_dataset[:util.TRAIN_COUNT],
            canny_edges[:util.TRAIN_COUNT]
        ))

        test_feature_matrix = np.hstack((
            hist_data[util.TRAIN_COUNT:],
            pca_dataset[util.TRAIN_COUNT:],
            canny_edges[util.TRAIN_COUNT:]
        ))
        #print('Training feature matrix dimensionality '+ str(training_feature_matrix.shape))

        if ESTIMATOR_POOL['svm'] is True:
            classifiers.svm(
                training_feature_matrix, training_targets, test_feature_matrix)
        if ESTIMATOR_POOL['knn'] is True:
            classifiers.knn(
                training_feature_matrix, training_targets, test_feature_matrix)
        if ESTIMATOR_POOL['random_forest'] is True:
            classifiers.random_forest(
                training_feature_matrix, training_targets, test_feature_matrix)
        if ESTIMATOR_POOL['adaboost'] is True:
            classifiers.adaboost(
                training_feature_matrix, training_targets, test_feature_matrix)
        if ESTIMATOR_POOL['extra_trees_classifier'] is True:
            classifiers.extra_trees_classifier(
                training_feature_matrix, training_targets, test_feature_matrix)
    
if __name__ == '__main__':
    main()
