#!/usr/bin/env python
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class PartitionedHistograms(BaseEstimator, TransformerMixin):

    def __init__(self, partitions=(3, 3, 3), interval=(1, 2000), bins=50):
        self.interval = interval
        self.bins = bins
        self.partitions = partitions

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        feature_count = (
            self.partitions[0]*self.partitions[1]*self.partitions[2]*self.bins
        )

        histograms = np.empty((data.shape[0], feature_count))

        for i, mri in enumerate(data):
            partitions = partition(mri, partitions=self.partitions)

            mri_histograms = [
                histogram(part, bins=self.bins, interval=self.interval)
                for part in partitions
            ]

            histograms[i] = np.array(mri_histograms).flatten()

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
