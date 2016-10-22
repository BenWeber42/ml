#!/usr/bin/env python
import numpy as np
import util


if __name__ == '__main__':
    train_set = util.load_all_train()
    test_set = util.load_all_test()
    components = np.load('./pca_1000_components.npy')

    train_set = util.nonzero_rv(train_set)
    test_set = util.nonzero_rv(test_set)

    train_set -= np.mean(train_set, axis=1, keepdims=True)
    test_set -= np.mean(test_set, axis=1, keepdims=True)

    train_reduced = np.dot(components, train_set)
    test_reduced = np.dot(components, test_set)

    np.save('train_reduced.npy', train_reduced)
    np.save('test_reduced.npy', test_reduced)
