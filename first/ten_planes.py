#!/usr/bin/env python
import numpy as np
import util
import matplotlib.pyplot as plt


REDUCED_DIM = 50
PLOTS_PATH = util.PLOTS_PATH + '/10_planes'
PLANES_Z = [20, 35, 56, 70, 87, 97, 107, 117, 127, 137]


def load_10_planes():

    planes = []
    for i in range(len(PLANES_Z)):
        z = PLANES_Z[i]
        print('Loading plane %d/%d (z = %d)' % (i + 1, len(PLANES_Z), z))
        planes.append(util.nonzero_rv(util.load_all_z_planes(z)))

    return np.concatenate(planes, axis=0)


if __name__ == '__main__':
    planes = load_10_planes()

    pc, _ = util.dense_pca(planes)
    reduced_planes = np.dot(pc[:, :REDUCED_DIM].T, planes)
    bias_data = np.concatenate([
        reduced_planes, np.ones((1, util.TRAIN_COUNT))
    ])

    refs = util.load_refs()

    # FIXME: replace constant bias with per dimension bias
    # for proper linear regression
    u, _, _, _ = np.linalg.lstsq(bias_data.T, refs.T)
    preds = np.dot(u.T, bias_data)

    plt.plot(preds, refs, 'yo', refs, refs, '--k')
    plt.savefig('%s/linear_regression50.png' % PLOTS_PATH)
    plt.close()

    plt.plot(u)
    plt.savefig('%s/linear_regression50_components.png' % PLOTS_PATH)
    plt.close()

    util.pca_plots(planes, PLOTS_PATH, components_correlation=5)
