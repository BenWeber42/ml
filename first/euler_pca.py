#!/usr/bin/env python
import numpy as np
import util


REDUCED_DIM = 1000
USE_FULL_DATA = True


if __name__ == '__main__':

    data = util.load_full_dataset()
    data = util.nonzero_rv(data)

    if USE_FULL_DATA:
        pca_data = data
    else:
        data -= np.mean(data, axis=1, keepdims=True)
        pca_data = data[:, :util.TRAIN_COUNT]

    pc, pv, pvr = util.truncated_pca(pca_data, n_components=REDUCED_DIM)

    data_reduced = np.dot(pc.T, data)

    if USE_FULL_DATA:
        np.save('%s/train_full_reduced.npy' % util.DATA_PATH,
                data_reduced[:, :util.TRAIN_COUNT])
        np.save('%s/test_full_reduced.npy' % util.DATA_PATH,
                data_reduced[:, util.TRAIN_COUNT:])
        np.save('%s/pca_full_%d_components.npy' % (util.DATA_PATH, REDUCED_DIM),
                pc)
    else:
        np.save('%s/train_reduced.npy' % util.DATA_PATH,
                data_reduced[:, :util.TRAIN_COUNT])
        np.save('%s/test_reduced.npy' % util.DATA_PATH,
                data_reduced[:, util.TRAIN_COUNT:])
        np.save('%s/pca_%d_components.npy' % (util.DATA_PATH, REDUCED_DIM),
                pc)
