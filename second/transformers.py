from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


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
            "DataDictExtractor: Key '%s' doesn't exist in data dict!" % self.key
        )

        return data_dict[self.key]


class ElementWiseTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, transformer):
        self.transformer = transformer

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        return np.array([
            np.array([
                self.transformer.transform([element]) for element in instance
            ]).flatten()
            for instance in data
        ])
