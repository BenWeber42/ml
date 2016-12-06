import numpy as np
import util
import histograms
import cv2


def detectEdges(volume, settings=(1, 1, 1)):

    maxValue = np.max(volume)

    mri = np.zeros(util.DIMS, dtype=np.float64)

    lowThreshold = 150
    highThreshold = 200

    if settings[0] == 1:
        for x in range(util.X_DIM):
            img = volume[x, :, :]
            cvuint8 = cv2.convertScaleAbs(img, 255.0/maxValue)
            edges = cv2.Canny(cvuint8, lowThreshold, highThreshold)
            mri[x, :, :] += edges

    if settings[1] == 1:
        for y in range(util.Y_DIM):
            img = volume[:, y, :]
            cvuint8 = cv2.convertScaleAbs(img, 255.0/maxValue)
            edges = cv2.Canny(cvuint8, lowThreshold, highThreshold)
            mri[:, y, :] += edges

    if settings[2] == 1:
        for z in range(util.Z_DIM):
            img = volume[:, :, z]
            cvuint8 = cv2.convertScaleAbs(img, 255.0/maxValue)
            edges = cv2.Canny(cvuint8, lowThreshold, highThreshold)
            mri[:, :, z] += edges

    # edge should be detected on two different slices containing the point
    # (x, y, z)  (e.g. in plane xy and plane yz that contains the point
    # (x, y, z))
    mri2 = mri > (256)
    mri2 = mri2.astype(int)

    return mri2


def generateEdgeFeaturesVector(
    volume,
    detectorSettings=(1, 1, 1),
    partitions=(9, 9, 9)
):
    mri = detectEdges(volume, detectorSettings)
    mri_partitioner = histograms.UniformPartitioner(partitions)
    partitions = mri_partitioner.get_parts(mri)

    features = []
    for partition in partitions:
        features.append(np.count_nonzero(partition)/partition.size)
    return features


def generateFullTrainMatrix(detectorSettings=(1, 1, 1), partitions=(9, 9, 9)):
    trainMatrix = []
    for i in range(0, util.TRAIN_COUNT-1):
        trainMatrix.append(generateEdgeFeaturesVector(util.load_train(i)))
    return trainMatrix


def generateFullTestMatrix(detectorSettings=(1, 1, 1), partitions=(9, 9, 9)):
    testMatrix = []
    for i in range(0, util.TEST_COUNT-1):
        testMatrix.append(generateEdgeFeaturesVector(util.load_test(i)))
    return testMatrix


def generateAndSaveFull():
    fullTrainMatrix = generateFullTrainMatrix()
    np.save("data\\train_full_edge", fullTrainMatrix)
    print("train_full_edge saved", np.shape(fullTrainMatrix))

    fullTestMatrix = generateFullTestMatrix()
    np.save("data\\test_full_edge", fullTestMatrix)
    print("test_full_edge saved", np.shape(fullTestMatrix))
