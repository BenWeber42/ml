import SimpleITK as sitk
import numpy as np
import util
from histograms import partition


def partitioned_canny_edges(
    data,
    observations_axis=0,
    partitions=(9, 9, 9),
    lower=200,
    upper=300
):

    if observations_axis == 1:
        data = data.T

    feature_count = partitions[0]*partitions[1]*partitions[2]
    samples_count = data.shape[0]

    edges = np.empty((samples_count, feature_count))

    for i, mri in enumerate(data):
        mri = mri.reshape(util.DIMS)

        mri_edges = canny_edges(mri, lower=lower, upper=upper)
        parts = partition(mri_edges, partitions=partitions)

        counts = [
            np.sum(part)
            for part in parts
        ]

        edges[i] = np.array(counts).flatten()

    if observations_axis == 1:
        edges = edges.T

    return edges


def canny_edges(mri, lower=200, upper=300):
    smri = sitk.GetImageFromArray(mri)
    sedges = sitk.CannyEdgeDetection(
        smri,
        lowerThreshold=lower,
        upperThreshold=upper
    )
    edges = sitk.GetArrayFromImage(sedges)
    return edges
