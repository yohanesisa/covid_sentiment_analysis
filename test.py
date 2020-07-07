import os
import time
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from Model.result import Result
from Module.helper import *

def doClassifier(kernel, classifier, training_feature, training_label, testing_feature, testing_label, c, tol, gamma='-', a='-', r='-'):
    classifier.fit(training_feature, training_label)

    prediction = classifier.predict(testing_feature)

    accuracy = accuracy_score(testing_label, prediction)
    precission = precision_score(testing_label, prediction, average='weighted')
    recall = recall_score(testing_label, prediction, average='weighted')
    f = f1_score(testing_label, prediction, average='weighted')
    c_matrix = confusion_matrix(testing_label, prediction, labels=[1, 0, -1])

    return Result(kernel=kernel, C=C, tol=tol, gamma=gamma, a=a, r=r, pos_true='-', pos_pred='-', net_true='-', net_pred='-', neg_true='-', neg_pred='-', accuracy_score=accuracy, precision_score=precission, recall_score=recall, f_score=f, confusion_matrix=c_matrix)

Cs          = [0.0078125,0.015625,0.03125,0.0625,0.125,0.25,0.5,1,2,4]
tols        = [0.0001,0.001,0.01,0.1,1]

gammas      = [0.0078125,0.015625,0.03125,0.0625,0.125,0.25,0.5,1,2,4]

aa          = [0.0078125,0.015625,0.03125,0.0625,0.125,0.25,0.5,1,2,4]
rr          = [0.0078125,0.015625,0.03125,0.0625,0.125,0.25,0.5,1,2,4]

max_passes  = -1

print '\n----------   List of Parameter   ----------'
print ''
print 'C          : ', len(Cs), ' ', str(Cs)
print 'tols       : ', len(tols), ' ', str(tols)
print ''
print 'gamma      : ', len(gammas), ' ', str(gammas)
print ''
print 'a          : ', len(aa), ' ', str(aa)
print 'r          : ', len(rr), ' ', str(rr)
print ''

FILE_TRAINING = 'test/training.xlsx'
FILE_TESTING = 'test/testing.xlsx'

# True
# False
feature_punc = True
feature_sentscore = True
feature_postag = True  
feature_unigram_tfidf = True

dual_class = False

FILE_EXPORT = ''

if feature_punc:
    FILE_EXPORT += 'Punc '
if feature_sentscore:
    FILE_EXPORT += 'Sent '
if feature_postag:
    FILE_EXPORT += 'PosTag '
if feature_unigram_tfidf:
    FILE_EXPORT += 'TFIDF '

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

timeObj = time.localtime(time.time())
timestamp = '%d-%d-%d %d:%d:%d' % (timeObj.tm_mday, timeObj.tm_mon, timeObj.tm_year, timeObj.tm_hour, timeObj.tm_min, timeObj.tm_sec)

# for index in range(len(training_label)):
#     print training_label[index]

# print '\n'

# for index in range(len(testing_label)):
#     print testing_label[index]

print '\nCalculating linear...'
result = []
for C in Cs:
    for tol in tols:
        classifier = svm.SVC(kernel='linear', C=C, tol=tol, max_iter=max_passes, decision_function_shape='ovr').fit(training_feature, training_label)

        result.append(doClassifier('linear', classifier, training_feature, training_label, testing_feature, testing_label, C, tol))

convertResultToDataFrame(result).to_excel('test/result/'+timestamp+' '+'linear.xlsx', index=False) 
print 'Exported to test/result/'+timestamp+' '+'linear.xlsx'




print '\nCalculating polynomial...'
result = []
for C in Cs:
    for tol in tols:
        classifier = svm.SVC(kernel='poly', coef0=1, degree=2, C=C, tol=tol, max_iter=max_passes, decision_function_shape='ovr').fit(training_feature, training_label)

        result.append(doClassifier('polynomial', classifier, training_feature, training_label, testing_feature, testing_label, C, tol))

convertResultToDataFrame(result).to_excel('test/result/'+timestamp+' '+'polynomial.xlsx', index=False) 
print 'Exported to test/result/'+timestamp+' '+'polynomial.xlsx'




print '\nCalculating rbf...'
result = []
for C in Cs:
    for tol in tols:
        for gamma in gammas:
            classifier = svm.SVC(kernel='rbf', gamma=gamma, C=C, tol=tol, max_iter=max_passes, decision_function_shape='ovr').fit(training_feature, training_label)

            result.append(doClassifier('rbf', classifier, training_feature, training_label, testing_feature, testing_label, C, tol, gamma=gamma))

convertResultToDataFrame(result).to_excel('test/result/'+timestamp+' '+'rbf.xlsx', index=False) 
print 'Exported to test/result/'+timestamp+' '+'rbf.xlsx'




print '\nCalculating sigmoid...'
result = []
for C in Cs:
    for tol in tols:
        for a in aa:
            for r in rr:
                classifier = svm.SVC(kernel='sigmoid', gamma=a, coef0=r, C=C, tol=tol, max_iter=max_passes, decision_function_shape='ovr').fit(training_feature, training_label)

                result.append(doClassifier('sigmoid', classifier, training_feature, training_label, testing_feature, testing_label, C, tol, a=a, r=r))

convertResultToDataFrame(result).to_excel('test/result/'+timestamp+' '+'sigmoid.xlsx', index=False) 
print 'Exported to test/result/'+timestamp+' '+'sigmoid.xlsx'

os.system('say "calculating with library svm finished"')