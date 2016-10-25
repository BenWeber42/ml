import numpy as np
import matplotlib.pyplot as plt
from nibabel.nifti1 import (
    load as _load_nifti1,
    save as _save_nifti1,
    Nifti1Image
)
from os.path import dirname, abspath
import csv


DATA_PATH = dirname(abspath(__file__)) + '/../data'
PLOTS_PATH = dirname(abspath(__file__)) + '/../plots'

DIMS = X_DIM, Y_DIM, Z_DIM = 176, 208, 176
TRAIN_COUNT = 278
TEST_COUNT = 138

FEATURE_SPACE = X_DIM * Y_DIM * Z_DIM
Z_PLANE_FEATURE_SPACE = X_DIM * Y_DIM


def load_refs():
    return np.asarray([int(age) for age in open('../data/targets.csv')])


def load_nifti1(path):
    return _load_nifti1(path).get_data().reshape(*DIMS)


def load_train(i):
    return load_nifti1('%s/set_train/train_%d.nii' % (DATA_PATH, i + 1))

def load_test(i):
    return load_nifti1('%s/set_test/test_%d.nii' % (DATA_PATH, i + 1))

def save(data, filename):
    _save_nifti1(Nifti1Image(data, None), filename)


def load_dataset(dims, load, observations_axis=1, dtype=np.float64):
    x_dim, y_dim = dims

    if observations_axis == 1:
        data = np.empty([x_dim, y_dim], dtype=dtype)
    else:
        data = np.empty([y_dim, x_dim], dtype=dtype)

    for num in range(y_dim):
        if observations_axis == 1:
            data[:, num] = load(num).reshape(x_dim)
        else:
            data[num, :] = load(num).reshape(x_dim)

    return data

def load_all_train(observations_axis=1, dtype=np.float64):
    return load_dataset(
            (FEATURE_SPACE, TRAIN_COUNT), load_train,
            observations_axis=observations_axis, dtype=dtype
        )

def load_all_test(observations_axis=1, dtype=np.float64):
    return load_dataset(
            (FEATURE_SPACE, TEST_COUNT), load_test,
            observations_axis=observations_axis, dtype=dtype
        )

def load_z_plane(i, z=int(Z_DIM/2), train_data=True):
    if train_data == True:
        return load_train(i)[:, :, z]
    else:
        return load_test(i)[:, :, z]


def load_all_z_planes(
    z=int(Z_DIM/2),
    train_data=True,
    observations_axis=1,
    dtype=np.float64
):

    def _load_z_plane(i):
        return load_z_plane(i, z, train_data)

    if train_data == True:
        return load_dataset(
                (Z_PLANE_FEATURE_SPACE, TRAIN_COUNT), _load_z_plane,
                observations_axis=observations_axis, dtype=dtype
            )
    else:
        return load_dataset(
                (Z_PLANE_FEATURE_SPACE, TEST_COUNT), _load_z_plane,
                observations_axis=observations_axis, dtype=dtype
            )

def nonzero_rv(data, observations_axis=1):
    nnz = np.nonzero(np.sum(data, axis=observations_axis))[0]

    if observations_axis == 1:
        return data[nnz, :]
    else:
        return data[:, nnz]


def dense_pca(X, full_matrices=False, observations_axis=1):
    X -= np.mean(X, axis=observations_axis, keepdims=True)

    print('Doing SVD on %d, %d matrix' % X.shape)
    if observations_axis == 1:
        u, d, _ = np.linalg.svd(X, full_matrices=full_matrices)
    else:
        u, d, _ = np.linalg.svd(X.T, full_matrices=full_matrices)

    return u, d**2


def pca_plots(data, plots_dir, components_correlation=10):
    pc, pv = dense_pca(data)

    data_reduced = np.dot(pc[:, :max(2, components_correlation)].T, data)

    refs = load_refs()

    plt.scatter(x=data_reduced[0], y=data_reduced[1],
                c=refs, cmap=plt.cm.hot)
    plt.colorbar()
    plt.savefig('%s/2d_pca.png' % plots_dir)
    plt.close()

    plt.plot(pv)
    plt.savefig('%s/eigen_spectrum.png' % plots_dir)
    plt.close()

    for component in range(components_correlation):
        linear_regression = np.poly1d(np.polyfit(
            data_reduced[component], refs, 1
        ))
        predictions = linear_regression(data_reduced[component])
        plt.plot(data_reduced[component], refs, 'yo',
                 data_reduced[component], predictions, '--k')
        plt.savefig('%s/component%d_correlation' % (plots_dir, component))
        plt.close()

def create_submission_file(labels, output_file='submission.csv'):
    with open('%s/%s' % (DATA_PATH, output_file), 'w') as outcsv:
        writer = csv.writer(outcsv)
        writer.writerow(["ID", "Prediction"])

        for row_data in zip(range(1, TEST_COUNT+1), labels):
            writer.writerow(list(row_data))
            
def load_preprocessed_data(data_files):
	# data_files is a tuple of training and test files which are
	# preprocessed and saved in .npy format.
	train_data = np.load('%s/%s' % (DATA_PATH, data_files[0]))
	test_data = np.load('%s/%s' % (DATA_PATH, data_files[1]))
	return (train_data, test_data)
