import numpy as np
import util


class NoPartitioner(object):

    def get_count(self):
        return 1

    def get_parts(self, mri, mask=None):
        return mri


class UniformPartitioner(object):

    def __init__(self, partitions=(3, 3, 3)):
        self.partitions = partitions

    def get_count(self):
        return self.partitions[0]*self.partitions[1]*self.partitions[2]

    def get_parts(self, mri, mask=None):
        mri = util.recover_from_masked(mri, mask)
        return [
            z_part
            for x_part in np.array_split(mri, self.partitions[0], axis=0)
            for y_part in np.array_split(x_part, self.partitions[1], axis=1)
            for z_part in np.array_split(y_part, self.partitions[2], axis=2)
        ]


class UniformHistogram(object):

    def __init__(self, interval=(1, 2000), bins=50, partitioner=NoPartitioner):
        self.interval = interval
        self.bins = 50
        self.partitioner = partitioner

    def compute_histogram(self, mri, mask=None):
        return np.concatenate([
            np.histogram(part, bins=self.bins, range=self.interval)[0]
            for part in
                self.partitioner.get_parts(mri, mask=mask)
        ])

    def compute_dataset_histograms(self, X, mask=None, observations_axis=1):
        if observations_axis == 1:
            X = X.T

        histograms = np.empty(
            (X.shape[0], self.partitioner.get_count()*self.bins)
        )

        for i in range(X.shape[0]):
            histograms[i] = self.compute_histogram(X[i], mask=mask)

        if observations_axis == 1:
            histograms = histograms.T

        return histograms
