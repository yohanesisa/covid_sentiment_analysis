import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.datasets.base import load_breast_cancer
from sklearn.manifold.t_sne import TSNE
from sklearn import svm

FILE_TRAINING = 'test/training_A.xlsx'
FILE_TESTING = 'test/testing_A.xlsx'

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

# testing_label = []
# testing_feature = []
# for testing_item in pd.read_excel(FILE_TESTING).values.tolist():
#     if dual_class:
#         if int(testing_item[0]) != 0:
#             testing_label.append(int(testing_item[0]))
            
#             temp_feature = []
#             if feature_punc:
#                 temp_feature.extend(testing_item[1:4])

#             if feature_sentscore:
#                 temp_feature.extend(testing_item[4:6])

#             if feature_postag:
#                 temp_feature.extend(testing_item[6:32])

#             if feature_unigram_tfidf:
#                 temp_feature.extend(testing_item[32:])

#             testing_feature.append(temp_feature)
#     else:
#         testing_label.append(int(testing_item[0]))
            
#         temp_feature = []
#         if feature_punc:
#             temp_feature.extend(testing_item[1:4])

#         if feature_sentscore:
#             temp_feature.extend(testing_item[4:6])

#         if feature_postag:
#             temp_feature.extend(testing_item[6:32])

#         if feature_unigram_tfidf:
#             temp_feature.extend(testing_item[32:])

#         testing_feature.append(temp_feature)

X_Train_embedded = TSNE(n_components=2).fit_transform(training_feature)

# Best Linear
# lin_c     = 0.03125
# lin_tol   = 0.0001

# pol_c     = 0.03125
# pol_tol   = 0.0001

# rbf_c     = 0.03125
# rbf_tol   = 0.0001
# rbf_gamma = 0.5

# sig_c     = 0.03125
# sig_tol   = 0.0001
# sig_a     = 0.125
# sig_r     = 0.03125

# Best Polynomial
# lin_c     = 0.0625
# lin_tol   = 0.001

# pol_c     = 0.0625
# pol_tol   = 0.001

# rbf_c     = 0.0625
# rbf_tol   = 0.001
# rbf_gamma = 0.5

# sig_c     = 0.0625
# sig_tol   = 0.001
# sig_a     = 0.0625
# sig_r     = 0.123

# Best RBF
# lin_c     = 0.25
# lin_tol   = 0.01

# pol_c     = 0.25
# pol_tol   = 0.01

# rbf_c     = 0.25
# rbf_tol   = 0.01
# rbf_gamma = 0.5

# sig_c     = 0.25
# sig_tol   = 0.01
# sig_a     = 0.25
# sig_r     = 2

# Best sigmoid
lin_c     = 1
lin_tol   = 0.01

pol_c     = 1
pol_tol   = 0.01

rbf_c     = 1
rbf_tol   = 0.01
rbf_gamma = 1

sig_c     = 1
sig_tol   = 0.01
sig_a     = 0.015625
sig_r     = 0.03125

models = (svm.SVC(kernel='linear', C=lin_c, tol=lin_tol, decision_function_shape='ovr'),
          svm.SVC(kernel='poly', coef0=1, degree=2, C=pol_c, tol=pol_tol, decision_function_shape='ovr'),
          svm.SVC(kernel='rbf', C=rbf_c, tol=rbf_tol, gamma=rbf_gamma, decision_function_shape='ovr'),
          svm.SVC(kernel='sigmoid', C=sig_c, tol=sig_tol, gamma=sig_a, coef0=sig_r, decision_function_shape='ovr'))
          
models = (clf.fit(X_Train_embedded, training_label) for clf in models)

# title for the plots
titles = ('Kernel Linear','Kernel Polynomial','Kernel RBF','Kernel Sigmoid')

# Set-up 2x2 grid for plotting.
fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = X_Train_embedded[:, 0], X_Train_embedded[:, 1]
xx, yy = make_meshgrid(X0, X1)

for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=training_label,
               cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()


# resolution = 1000 # 100x100 background pixels
# plt.scatter(X_Train_embedded[:,0], X_Train_embedded[:,1], c=training_label)
# plt.show()