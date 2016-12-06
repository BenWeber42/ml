#!/usr/bin/env python
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from pipeline import (
    DataPipeline,
    FeatureExtractors
)


if __name__ == '__main__':

    print('Creating Data Pipeline')
    data_pipeline = DataPipeline()
    print('Creating Feature Extractors Pipeline')
    feature_extractors = FeatureExtractors(data_pipeline, {
        FeatureExtractors.HISTOGRAMS: {
            'partitions': [(9, 9, 9), (3, 3, 3), (6, 6, 6)],
            'bins': [15, 45, 100],
            'interval': [(0, 2000), (1, 2000)]
        },
        FeatureExtractors.PCA: {}
    })

    print('Creating Grid Search CV Pipeline')
    pipeline = Pipeline([
        ('data_dict_builder', data_pipeline.data_dict_builder),
        ('feature_extraction', feature_extractors.get_pipeline()),
        ('linear_svm', SVC(kernel='linear', probability=True))
    ])

    param_grid = {
        'linear_svm__C': [0.025]
    }

    param_grid.update(feature_extractors.get_tuning_parameters(
        'feature_extraction__'))

    optimizer = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring='neg_log_loss',
        cv=10
    )

    print('Tuning hyperparameters')
    optimizer.fit(data_pipeline.train_data, data_pipeline.refs)

    for params, mean_score, scores in optimizer.grid_scores_:
        print("Mean Score: %.5f, Stddev: %.5f for %r" % (
            mean_score, scores.std(), params
        ))
