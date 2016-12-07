#!/usr/bin/env python
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import util


class PartitionedHistograms(BaseEstimator, TransformerMixin):

    def __init__(self, partitions=(3, 3, 3), interval=(1, 2000), bins=50):
        self.interval = interval
        self.bins = bins
        self.partitions = partitions

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        return partitioned_histograms(
            data,
            observations_axis=0,
            partitions=self.partitions,
            interval=self.interval,
            bins=self.bins
        )


def partitioned_histograms(
    data,
    observations_axis=0,
    partitions=(3, 3, 3),
    interval=(1, 2000),
    bins=50
):
    if observations_axis == 1:
        data = data.T

    feature_count = partitions[0]*partitions[1]*partitions[2]*bins
    samples_count = data.shape[0]

    histograms = np.empty((samples_count, feature_count))

    for i, mri in enumerate(data):
        mri = mri.reshape(util.DIMS)
        partitions = partition(mri, partitions=partitions)

        mri_histograms = [
            histogram(part, bins=bins, interval=interval)
            for part in partitions
        ]

        histograms[i] = np.array(mri_histograms).flatten()

    if observations_axis == 1:
        histograms = histograms.T

    return histograms


def partition(mri, partitions=(3, 3, 3)):
    return [
        z_part
        for x_part in np.array_split(mri, partitions[0], axis=0)
        for y_part in np.array_split(x_part, partitions[1], axis=1)
        for z_part in np.array_split(y_part, partitions[2], axis=2)
    ]


def histogram(data, interval=(1, 2000), bins=50):
    return np.histogram(data, bins=bins, range=interval)[0]
