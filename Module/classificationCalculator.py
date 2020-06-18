import copy
import pandas as pd
from Model.result import Result
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

def svmClassification(training_model, testing_kernel, kernel_type):
    y_true = []
    y_pred = []

    for testing_index, K in enumerate(testing_kernel):
        result_OAA = { 1: 0.0, 0: 0.0, -1: 0.0 }

        OAA = [1,0,-1]
        for index, pov in enumerate(OAA):
            a = training_model[index].getAlpha()
            b = training_model[index].getBias()

            y = copy.deepcopy(training_model[index].getLabel())
            for index_y, item in enumerate(y):
                if item == pov:
                    y[index_y] = 1
                else:
                    y[index_y] = -1

            for i in range(1,len(K)):
                result_OAA[pov] += (float(a[i-1])*float(y[i-1])*float(K[i]))

            result_OAA[pov] += float(b)

        y_true.append(int(K[0]))
        y_pred.append(max(result_OAA, key=result_OAA.get))

    pos_true = y_true.count(1)
    net_true = y_true.count(0)
    neg_true = y_true.count(-1)

    pos_pred = y_pred.count(1)
    net_pred = y_pred.count(0)
    neg_pred = y_pred.count(-1)

    accuracy = accuracy_score(y_true, y_pred)
    precission = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f = f1_score(y_true, y_pred, average='weighted')

    return Result(kernel=kernel_type, C=training_model[0].getC(), tol=training_model[0].getTol(), gamma=training_model[0].getGamma(), a=training_model[0].getA(), r=training_model[0].getR(), pos_true=pos_true, pos_pred=pos_pred, net_true=net_true, net_pred=net_pred, neg_true=neg_true, neg_pred=neg_pred, accuracy_score=accuracy, precision_score=precission, recall_score=recall, f_score=f)
