#!/usr/bin/env python -W ignore::DeprecationWarning
import numpy as np
import util
from sklearn import preprocessing

PLANES_Z = [20, 35, 56, 70, 87, 97, 107, 117, 127, 137]
#PLANES_Z = list(range(20,170)) # Load all planes.

PRINT_ESTIMATOR_RESULTS = False # If True, prints results for all possible configurations.
N_JOBS = 6 # Num of CPU cores for parallel processing.

def load_z_planes():
    training_planes = []
    for i in range(len(PLANES_Z)):
        z = PLANES_Z[i]
        print('Loading training plane %d/%d (z = %d)' % (i + 1, len(PLANES_Z), z))
        training_planes.append(util.load_all_z_planes(z))

    test_planes = []
    for i in range(len(PLANES_Z)):
        z = PLANES_Z[i]
        print('Loading test plane %d/%d (z = %d)' % (i + 1, len(PLANES_Z), z))
        test_planes.append(util.load_all_z_planes(z, False))

    return (np.concatenate(training_planes, axis=0), np.concatenate(test_planes, axis=0))

def main():
    ####################################
    # Prepare feature matrices.
    ####################################
    normalizer = preprocessing.StandardScaler()
    training_targets = util.load_refs() # Load targets
    # Load train & test data.
    print('Loading and preprocessing raw data.')
    training_planes, test_planes = load_z_planes()
    training_planes = training_planes.T
    test_planes = test_planes.T
    all_planes = np.vstack((training_planes, test_planes))

    print('Data dimensionality ' + str(all_planes.shape))

    # TODO: Feature pool will come here


if __name__ == '__main__':
    main()