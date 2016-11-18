#!/usr/bin/env python
import numpy as np
import util


REDUCED_DIM = 1000

if __name__ == '__main__':

    data, _ = util.load_full_nnz_dataset()

    pc, _, _ = util.truncated_pca(data, n_components=REDUCED_DIM)

    data_reduced = np.dot(pc.T, data)

    np.save('%s/train_full_pca.npy' % util.DATA_PATH,
            data_reduced[:, :util.TRAIN_COUNT])
    np.save('%s/test_full_pca.npy' % util.DATA_PATH,
            data_reduced[:, util.TRAIN_COUNT:])
