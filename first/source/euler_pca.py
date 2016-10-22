#!/usr/bin/env python
import numpy as np
from sklearn.decomposition import TruncatedSVD
import util


REDUCED_DIM = 1000


if __name__ == '__main__':
    with open('euler_pca.log', 'w') as logfile:
        logfile.write('loading data ... ')
        data = util.load_all_train()
        logfile.write('DONE\n')

        logfile.write('removing zeros from data ... ')
        data = util.nonzero_rv(data)
        logfile.write('DONE\n')

        logfile.write('mean centering data ... ')
        data -= np.mean(data, axis=1, keepdims=True)
        logfile.write('DONE\n')

        logfile.write('creating SVD object ... ')
        tsvd = TruncatedSVD(n_components=1000)
        logfile.write('DONE\n')

        logfile.write('applying truncated svd ... ')
        tsvd.fit(data.T)
        logfile.write('DONE\n')

        logfile.write('saving variances ... ')
        np.save('./variances.npy', tsvd.explained_variance_)
        logfile.write('DONE\n')

        logfile.write('saving variance ratios ... ')
        np.save('./variance_ratios.npy', tsvd.explained_variance_ratio_)
        logfile.write('DONE\n')

        logfile.write('saving components ... ')
        np.save('./pca_%d_components.npy' % REDUCED_DIM, tsvd.components_)
        logfile.write('DONE\n')
