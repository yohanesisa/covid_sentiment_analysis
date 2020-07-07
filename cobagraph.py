
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm, datasets


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

feature_punc = False
feature_sentscore = True
feature_postag = False
feature_unigram_tfidf = False

training_label = []
training_feature = []
for training_item in pd.read_excel('test/training.xlsx').values.tolist():
    training_label.append(int(training_item[0]))

    temp_feature = []
    if feature_punc:
        temp_feature.extend(training_item[1:3])

    if feature_sentscore:
        temp_feature.extend(training_item[0:2])

    if feature_postag:
        temp_feature.extend(training_item[7:33])

    if feature_unigram_tfidf:
        temp_feature.extend(training_item[34:])

    training_feature.append(temp_feature)

training_label = np.array(training_label)
training_feature = np.array(training_feature)


# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 2
tol = 2
models = (svm.SVC(kernel='linear', C=C, tol=tol, max_iter=5, decision_function_shape='ovr'),
          svm.SVC(kernel='poly', coef0=1, degree=2, C=C, tol=tol, max_iter=-1, decision_function_shape='ovr'),
          svm.SVC(kernel='rbf', C=C),
          svm.SVC(kernel='sigmoid', C=C),)
#   svm.SVC(kernel='rbf', gamma=0.7, C=C),
#   svm.SVC(kernel='sigmoid', gamma=0.7, coef0=0.5, C=C),)
models = (clf.fit(training_feature, training_label) for clf in models)

# title for the plots
titles = ('Kernel Linear','Kernel Polynomial','Kernel RBF','Kernel Sigmoid')

# Set-up 2x2 grid for plotting.
fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = training_feature[:, 0], training_feature[:, 1]
xx, yy = make_meshgrid(X0, X1)

for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=training_label,
               cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Sentiment Score Positive')
    ax.set_ylabel('Sentiment Score Negative')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()
