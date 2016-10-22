This directory should contain the data. There should be a 'set_train' and 'set_test' folder
as well as a 'sampleSubmission.csv' and 'targets.csv'.

- train_reduced.npy & test_reduced.npy:
  Those data sets are dimensionality reduced data sets of the full (nonzero)
  train and test resp. data sets. A PCA has been carried out on the full
  (nonzero) training dataset and the first 1000 components were kept. Then
  both the training and test data sets were transformed using those 1000
  components.
