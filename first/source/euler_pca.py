#!/usr/bin/env python
import numpy as np
import util


REDUCED_DIM = 1000


if __name__ == '__main__':

    data = util.load_full_dataset()
    data = util.nonzero_rv(data)

    pc, pv, pvr = util.truncated_pca(data, n_components=REDUCED_DIM)

    data_reduced = np.dot(pc.T, data)

    np.save('%s/train_full_reduced.npy' % util.DATA_PATH,
            data_reduced[:util.TRAIN_COUNT])
    np.save('%s/test_full_reduced.npy' % util.DATA_PATH,
            data_reduced[util.TRAIN_COUNT:])
    np.save('%s/full_pca_%d_components.npy' % util.DATA_PATH,
            pc)
