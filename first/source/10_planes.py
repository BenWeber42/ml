#!/usr/bin/env python
import numpy as np
import util


REDUCED_DIM = 50
PLOTS_PATH = util.PLOTS_PATH + '/10_planes'
PLANES_Z = [20, 35, 56, 70, 87, 97, 107, 117, 127, 137]

if __name__ == '__main__':
    planes = []
    for i in range(len(PLANES_Z)):
        z = PLANES_Z[i]
        print('Loading plane %d/%d (z = %d)' % (i + 1, len(PLANES_Z), z))
        planes.append(util.nonzero_rv(util.load_all_z_planes(z)))

    data = np.concatenate(planes, axis=0)

    util.pca_plots(data, PLOTS_PATH, components_correlation=5)
