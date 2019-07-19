# Machine Learning Project

In this project, we were given a training dataset of ~200 3D MRI brain scans.
Throughout the three milestones, we had to predict gender, age & cognitive disabilities.

We used various features:

1) A principle component analysis (PCA) to reduce the dimensionality
2) Histograms of values of partitioned brains
3) Histograms of canny edges of partitioned brains

We tried out several ML methods from `sklearn` such as
random forests, SVM (kernelized) or KNN. We used cross validation
to tune the resp. hyper parameters.
