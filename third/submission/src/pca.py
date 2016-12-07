#!/usr/bin/env python
import numpy as np
import util


if __name__ == '__main__':

    data = util.load_full_dataset()

    pc, _ = util.dense_pca(data)

    data_reduced = np.dot(pc.T, data)

    np.save('%s/full_pca.npy' % util.ADDITIONAL_DATA_PATH, data_reduced)
