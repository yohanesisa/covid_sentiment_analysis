import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.datasets.base import load_breast_cancer
from sklearn.manifold.t_sne import TSNE
from sklearn import svm

FILE_TRAINING = 'test/training_A.xlsx'
FILE_TESTING = 'test/testing_A.xlsx'

# FILE_TRAINING = 'test/training_B.xlsx'
# FILE_TESTING = 'test/testing_B.xlsx'

# FILE_TRAINING = 'test/training_C.xlsx'
# FILE_TESTING = 'test/testing_C.xlsx'

# FILE_TRAINING = 'test/training_manual.xlsx'
# FILE_TESTING = 'test/testing_manual.xlsx'

# FILE_TRAINING = 'test/training_lexicon.xlsx'
# FILE_TESTING = 'test/testing_lexicon.xlsx'

feature_punc = True
feature_sentscore = True
feature_postag = True  
feature_unigram_tfidf = True

dual_class = False

def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

training_label = []
training_feature = []
for training_item in pd.read_excel(FILE_TRAINING).values.tolist():
    if dual_class:
        if int(training_item[0]) != 0:
            training_label.append(int(training_item[0]))
            
            temp_feature = []
            if feature_punc:
                temp_feature.extend(training_item[1:4])

            if feature_sentscore:
                temp_feature.extend(training_item[4:6])

            if feature_postag:
                temp_feature.extend(training_item[6:32])

            if feature_unigram_tfidf:
                temp_feature.extend(training_item[32:])

            training_feature.append(temp_feature)
    else:  
        training_label.append(int(training_item[0]))
        
        temp_feature = []
        if feature_punc:
            temp_feature.extend(training_item[1:4])

        if feature_sentscore:
            temp_feature.extend(training_item[4:6])

        if feature_postag:
            temp_feature.extend(training_item[6:32])

        if feature_unigram_tfidf:
            temp_feature.extend(training_item[32:])

        training_feature.append(temp_feature)

testing_label = []
testing_feature = []
for testing_item in pd.read_excel(FILE_TESTING).values.tolist():
    if dual_class:
        if int(testing_item[0]) != 0:
            testing_label.append(int(testing_item[0]))
            
            temp_feature = []
            if feature_punc:
                temp_feature.extend(testing_item[1:4])

            if feature_sentscore:
                temp_feature.extend(testing_item[4:6])

            if feature_postag:
                temp_feature.extend(testing_item[6:32])

            if feature_unigram_tfidf:
                temp_feature.extend(testing_item[32:])

            testing_feature.append(temp_feature)
    else:
        testing_label.append(int(testing_item[0]))
            
        temp_feature = []
        if feature_punc:
            temp_feature.extend(testing_item[1:4])

        if feature_sentscore:
            temp_feature.extend(testing_item[4:6])

        if feature_postag:
            temp_feature.extend(testing_item[6:32])

        if feature_unigram_tfidf:
            temp_feature.extend(testing_item[32:])

        testing_feature.append(temp_feature)

X_Train_embedded = TSNE(n_components=2).fit_transform(training_feature)
plt.scatter(X_Train_embedded[:,0], X_Train_embedded[:,1], c=training_label)
plt.show()

# X_Train_embedded = TSNE(n_components=2).fit_transform(testing_feature)
# plt.scatter(X_Train_embedded[:,0], X_Train_embedded[:,1], c=testing_label)
# plt.show()