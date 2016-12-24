#!/usr/bin/env python
import numpy as np
import util
from histograms import partitioned_histograms
from canny_edges import partitioned_canny_edges
import classifier_pool as classifiers
np.random.seed(1)

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
    dataset = util.load_full_dataset()
    pca_dataset = util.load_pca_dataset()

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

    print('Precomputing canny edges.')
    canny_edges = partitioned_canny_edges(dataset)

    for hist_data, hist_cfg in histograms:

        feature_cfg = {
            'histograms': hist_cfg,
            'canny_edges': 'default'
        }

        print('Using features settings:')
        print(feature_cfg)

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
