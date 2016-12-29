import SimpleITK as sitk
import numpy as np
import util
from histograms import partition

def partitioned_watersheds_segmentation(
    data,
    observations_axis=0,
    partitions=(9, 9, 9),
    level=4
):

    if observations_axis == 1:
        data = data.T

    feature_count = partitions[0]*partitions[1]*partitions[2]
    samples_count = data.shape[0]

    edges = np.empty((samples_count, feature_count))

    for i, mri in enumerate(data):
        mri = mri.reshape(util.DIMS)

        mri_watersheds = watersheds_segmentation(mri, level=level)
        parts = partition(mri_watersheds, partitions=partitions)

        counts = [
            np.sum(part)
            for part in parts
        ]

        watersheds[i] = np.array(counts).flatten()

    if observations_axis == 1:
        watersheds = watersheds.T

    return watersheds


def watersheds_segmentation(mri, level=4):
    smri = sitk.GetImageFromArray(mri)
    swatersheds = sitk.GradientMagnitude(smri)
    watersheds = sitk.MorphologicalWatershed(swatersheds, level=level, markWatershedLine=True, fullyConnected=False)
    #watersheds= sitk.LabelToRGB(watersheds)
    watersheds = sitk.GetArrayFromImage(watersheds)
    return watersheds