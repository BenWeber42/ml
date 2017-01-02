#!/usr/bin/env python
import numpy as np
import util
from sklearn import preprocessing, decomposition
from histograms import partitioned_histograms
from canny_edges import partitioned_canny_edges
import classifier_pool as classifiers
import time # For timestamp.
import datetime
np.random.seed(1)

timeForThisRun = time.time()
timeStamp = datetime.datetime.fromtimestamp(timeForThisRun).strftime('%m%d%H%M%S')
classifiers.SUBMISSION_FILE_PREFIX = timeStamp
print('------------------')
print(timeStamp)
print('------------------')

ESTIMATOR_POOL = {
    'random_forest': True,
    'knn': False,
    'extra_trees_classifier': False,
    'svm': False, # Does not support multi-label classification
    'adaboost': False, # Does not support multi-label classification
}

FEATURE_POOL = {
    'pca_disk': False,
    'pca': True,
    'histogram': True,
    'canny_edges': True,
}

PCA_REDUCED_DIM = 50

def main():
    ####################################
    # Prepare feature matrices.
    ####################################
    # If the corresponding feature extraction method is not used, then
    # this will be used as a replacement.
    emptyDataset = np.empty(shape=(util.TRAIN_COUNT+util.TEST_COUNT, 0))
    normalizer = preprocessing.StandardScaler()
    print('Loading Targets.')
    training_targets = util.load_refs()  # Load targets

    if FEATURE_POOL['pca_disk'] == True:
        print('Loading PCA.')
        pca_disk = util.load_pca_dataset()
        pca_disk_cfg = str(pca_disk.shape[1])
    else:
        pca_disk = emptyDataset
        pca_disk_cfg = 'None'

    if FEATURE_POOL['histogram'] == True or FEATURE_POOL['canny_edges'] == True or FEATURE_POOL['pca'] == True:
        print('Loading Data.')
        dataset = util.load_full_dataset()
        #dataset = ((dataset-dataset.min())/(dataset.max()-dataset.min()))*255

    if FEATURE_POOL['pca'] == True:
        print('Precomputing PCA.')
        datasetStandardized = normalizer.fit_transform(dataset)
        pca_dataset = decomposition.PCA(n_components=PCA_REDUCED_DIM, whiten=False).fit_transform(datasetStandardized)
        pca_cfg = str(PCA_REDUCED_DIM)
    else:
        pca_dataset = emptyDataset
        pca_cfg = 'None'

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
            ]
            for interval in [
                (1, 2000)
            ]
            for bins in [
                5
            ]
        ]
    else:
        histograms = [[emptyDataset, 'None']]

    if FEATURE_POOL['canny_edges'] == True:
        print('Precomputing canny edges.')
        canny_edges = [
            (
                partitioned_canny_edges(
                    dataset,
                    partitions=partitions,
                    lower=lower,
                    upper=300
                ),
                {
                    'partitions': partitions,
                    'lower': lower,
                    'upper': 300
                }
            )
            for partitions in [
                (9, 9, 9),
            ]
            for lower in [
                100
            ]
        ]
    else:
        canny_edges = [[emptyDataset, 'None']]

    # To better keep track of the submission files and output logs.
    print('Classification.')
    featureConfigurationID = 0
    for hist_data, hist_cfg in histograms:
        for canny_edge_data, canny_edges_cfg in canny_edges:
            featureConfigurationID = featureConfigurationID + 1
            classifiers.SUBMISSION_FILE_SUFFIX = '(' + str(featureConfigurationID) + ')'

            feature_cfg = {
                'pca_disk': pca_disk_cfg,
                'pca': pca_cfg,
                'histograms': hist_cfg,
                'canny_edges': canny_edges_cfg
            }

            training_feature_matrix = np.hstack((
                hist_data[:util.TRAIN_COUNT],
                pca_disk[:util.TRAIN_COUNT],
                pca_dataset[:util.TRAIN_COUNT],
                canny_edge_data[:util.TRAIN_COUNT]
            ))

            test_feature_matrix = np.hstack((
                hist_data[util.TRAIN_COUNT:],
                pca_disk[util.TRAIN_COUNT:],
                pca_dataset[util.TRAIN_COUNT:],
                canny_edge_data[util.TRAIN_COUNT:]
            ))
            print('---------------------------------------------')
            print(classifiers.SUBMISSION_FILE_SUFFIX + ' {' + str(training_feature_matrix.shape) + '} ' + str(feature_cfg))

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
