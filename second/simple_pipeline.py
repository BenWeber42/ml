#!/usr/bin/env python
from sklearn.pipeline import (
    Pipeline,
    FeatureUnion
)
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import util
from transformers import (
    DataDictBuilder,
    DataDictExtractor,
    ElementWiseTransformer
)
from histograms import PartitionedHistograms


from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


RAW_DATA = 'raw_data'
PCA_DATA = 'pca_data'


names = [
    "Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
    "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
    "Naive Bayes", "QDA"
]

classifiers = [
    KNeighborsClassifier(),
    SVC(kernel="linear", C=0.025, probability=True),
    SVC(gamma=2, C=1, probability=True),
    GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()
]


if __name__ == '__main__':

    print('Loading Data')
    raw_data, nnz = util.load_all_nnz_train(observations_axis=0)
    pca_data = util.load_full_pca_train(observations_axis=0)

    raw_test_data, _ = util.load_all_nnz_test(observations_axis=0)
    pca_test_data = util.load_full_pca_test(observations_axis=0)
    #data = np.hstack((raw_data, pca_data))

    refs = util.load_refs()

    print('Building Pipeline')
    histogramer = PartitionedHistograms(
        mask=nnz,
        bins=45,
        interval=(0, 2000),
        partitions=(9, 9, 9)
    )

#    classifier = SVC(probability=True)
#    classifier = KNeighborsClassifier()

#    pipeline = Pipeline([
#        ('data_dict_builder', DataDictBuilder({
#            RAW_DATA: (0, raw_data.shape[1]),
#            PCA_DATA: (raw_data.shape[1], raw_data.shape[1] + pca_data.shape[1])
#        })),
#        ('feature_extraction', FeatureUnion(transformer_list=[
#            ('histograms', Pipeline([
#                ('raw_data_extractor', DataDictExtractor(RAW_DATA)),
#                ('partitioned_histograms', histogramer),
#            ])),
#            ('pca', DataDictExtractor(PCA_DATA))
#        ])),
#        ('feature_selection', SelectKBest(k=200)),
#        ('classification', classifier),
#    ])

    print('Doing Feature Extraction')
    histograms = histogramer.fit(raw_data).transform(raw_data)
    test_histograms = histogramer.fit(raw_test_data).transform(raw_test_data)
    data = np.hstack((pca_data, histograms))
    test_data = np.hstack((pca_test_data, test_histograms))

    print('Doing Cross-validation')
    for name, classifier in zip(names, classifiers):

        scores = cross_val_score(classifier, data, refs, cv=10, scoring='neg_log_loss')
        print('%s: Avg score: %.5f Stddev: %.5f' % (
            name, scores.mean(), scores.std()))
        print('predictions')
        print(classifier.fit(data, refs).predict_proba(test_data))

