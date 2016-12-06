from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import (
    Pipeline,
    FeatureUnion
)

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import numpy as np

import util
from histograms import PartitionedHistograms


class DataDictBuilder(BaseEstimator, TransformerMixin):

    def __init__(self, dict):
        self.dict = dict

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        return {
            key: data[:, lower:upper]
            for key, (lower, upper) in self.dict.items()
        }


class DataDictExtractor(BaseEstimator, TransformerMixin):

    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        assert self.key in data_dict.keys(), (
            "DataDictExtractor: Key '%s' doesn't exist in data dict!"
            % self.key
        )

        return data_dict[self.key]


class DataPipeline(object):

    """
    Loads data and exposes accessibility fields for the data to be used
    in an sklearn pipeline

    Exposed fields

    nnz:                    nnz mask (to recover a 3D MRI)
    refs:                   reference solutions/targets/y
    train_data:             training raw dataset & training pca reduced dataset
    test_data:              test raw dataset & test pca reduced dataset
    data_dict_builder:      transforms data array to data dict with separate
                            components (use data extractors for the components)
    raw_data_extractor:     transfroms data dict to the raw dataset
    pca_data_extractor:     transforms data dict to the pca reduced dataset
    """

    def __init__(self):

        RAW_DATA = 'raw_data'
        PCA_DATA = 'pca_data'

        self.refs = util.load_refs()

        train_raw_data, self.nnz = util.load_all_nnz_train(observations_axis=0)
        train_pca_data = util.load_full_pca_train(observations_axis=0)

        test_raw_data, _ = util.load_all_nnz_test(observations_axis=0)
        test_pca_data = util.load_full_pca_test(observations_axis=0)

        self.train_data = np.hstack((train_raw_data, train_pca_data))
        self.test_data = np.hstack((test_raw_data, test_pca_data))
        self.all_data = np.hstack((
            np.vstack((train_raw_data, test_raw_data)),
            np.vstack((train_pca_data, test_pca_data)))
        )

        self.data_dict_builder = DataDictBuilder({
            RAW_DATA: (0, train_raw_data.shape[1]),
            PCA_DATA: (train_raw_data.shape[1], self.train_data.shape[1])
        })

        self.raw_data_extractor = DataDictExtractor(RAW_DATA)
        self.pca_data_extractor = DataDictExtractor(PCA_DATA)


class FeatureExtractors(object):

    """
    Creates configurable extractors pipeline

    Sample:

    feature_extractor = FeatureExtractors(data_pipeline, {
        FeatureExtractors.HISTOGRAMS: {
            'partitions': [(9, 9, 9), (3, 3, 3)],
            'bins': [20, 45, 100],
            'interval': [(0, 2000)]
        },
        FeatureExtractors.PCA: {}
    })

    Creates a FeatureUnion for each key, value pair in configs.
    Each feature extractor can be given tuning parameters through a dict.
    Each parameter is a list of possible values (the first one will be used
    as default value) which can be fetched through
    FeatureExtractors.get_tuning_parameters
    """

    HISTOGRAMS = 'histograms'
    PCA = 'pca'

    def __init__(self, data_pipeline, configs):
        self.data_pipeline = data_pipeline

        self.extractors = []
        self.tuning_parameters = {}
        for name, config in configs.items():
            if name == FeatureExtractors.HISTOGRAMS:
                self.add_histograms(config)
            elif name == FeatureExtractors.PCA:
                self.add_pca()
            else:
                print("FeatureExtractors: Unknown extractor '%s'!" % name)

        self.pipeline = FeatureUnion(transformer_list=self.extractors)

    def add_histograms(self, config):
        self.extractors.append((
            FeatureExtractors.HISTOGRAMS,
            Pipeline([
                ('raw_data_extractor', self.data_pipeline.raw_data_extractor),
                ('partitioned_histograms', PartitionedHistograms(
                    mask=self.data_pipeline.nnz,
                    **self.get_default_parameters(config)
                ))
            ])
        ))
        for p_name, parameters in config.items():
            self.tuning_parameters[
                FeatureExtractors.HISTOGRAMS +
                '__partitioned_histograms__' +
                p_name
            ] = parameters

    def add_pca(self):
        self.extractors.append((
            FeatureExtractors.PCA, self.data_pipeline.pca_data_extractor
        ))

    def get_default_parameters(self, config):
        return {
            name: tuning_parameters[0]
            for name, tuning_parameters in config.items()
        }

    def get_pipeline(self):
        return self.pipeline

    def get_tuning_parameters(self, prefix=''):
        return {
            prefix + name: values
            for name, values in self.tuning_parameters.items()
        }


classifiers = {
    "Nearest Neighbors": KNeighborsClassifier(),
    "Linear SVM": SVC(kernel="linear", C=0.025, probability=True),
    "RBF SVM": SVC(gamma=2, C=1, probability=True),
    "Gaussian Process":
        GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    "Decision Tree": DecisionTreeClassifier(max_depth=5),
    "Random Forest":
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    "Neural Net": MLPClassifier(alpha=1),
    "AdaBoost": AdaBoostClassifier(),
    "Naive Bayes": GaussianNB(),
    "QDA": QuadraticDiscriminantAnalysis(),
}

# TODO: configurable classifier pipline generator
