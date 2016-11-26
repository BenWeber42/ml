#!/usr/bin/env python
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import util


class PartitionedHistograms(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        partitions=(3, 3, 3),
        interval=(1, 2000),
        bins=50,
        mask=None
    ):
        self.interval = interval
        self.bins = bins
        self.partitions = partitions
        self.mask = mask

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        feature_count = (
            self.partitions[0]*self.partitions[1]*self.partitions[2]*self.bins
        )

        histograms = np.empty((data.shape[0], feature_count))

        for i, mri in enumerate(data):
            partitions = partition(
                mri,
                partitions=self.partitions,
                mask=self.mask
            )

            mri_histograms = [
                histogram(part, bins=self.bins, interval=self.interval)
                for part in partitions
            ]

            histograms[i] = np.array(mri_histograms).flatten()

        return histograms


def partition(mri, partitions=(3, 3, 3), mask=None):
    mri = util.recover_from_masked(mri, mask)

    return [
        z_part
        for x_part in np.array_split(mri, partitions[0], axis=0)
        for y_part in np.array_split(x_part, partitions[1], axis=1)
        for z_part in np.array_split(y_part, partitions[2], axis=2)
    ]


def histogram(data, interval=(1, 2000), bins=50):
    return np.histogram(data, bins=bins, range=interval)[0]


if __name__ == '__main__':
    train, nnz = util.load_all_nnz_train(observations_axis=0)
    test, _ = util.load_all_nnz_test(observations_axis=0)

    histogramer = PartitionedHistograms(
        partitions=(9, 9, 9),
        interval=(1, 2000),
        bins=45,
        mask=nnz
    )

    train_histograms = histogramer.fit(train).transform(train)
    test_histograms = histogramer.fit(test).transform(test)

    np.save('%s/train_histograms.npy' % util.DATA_PATH, train_histograms)
    np.save('%s/test_histograms.npy' % util.DATA_PATH, test_histograms)
