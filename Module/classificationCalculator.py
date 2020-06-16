import copy
import pandas as pd

def svmClassification(training_model, kernel_type):
    
    if kernel_type == 'linear':
        testing = pd.read_excel('Export/kernel/testing/linear.xlsx').values.tolist()
    elif kernel_type == 'polynomial':
        testing = pd.read_excel('Export/kernel/testing/polynomial.xlsx').values.tolist()
    elif kernel_type == 'rbf':
        testing = pd.read_excel('Export/kernel/testing/rbf.xlsx').values.tolist()
    elif kernel_type == 'sigmoid':
        testing = pd.read_excel('Export/kernel/testing/sigmoid.xlsx').values.tolist()

    result = { 1: 0, 0: 0, -1: 0 }
    true_classification = 0
    false_classification = 0

    for test in testing:
        print test

    for testing_index, K in enumerate(testing):
        result_OAA = { 1: 0.0, 0: 0.0, -1: 0.0 }

        OAA = [1, 0, -1]
        for index, pov in enumerate(OAA):
            a = training_model[index].getAlpha()
            b = training_model[index].getBias()

            y = copy.deepcopy(training_model[index].getLabel())
            for index_y, item in enumerate(y):
                if item == pov:
                    y[index_y] = 1
                else:
                    y[index_y] = -1

            # result_OAA = 0.0
            for i in range(1,len(K)):
                result_OAA[pov] += (float(a[i-1])*float(y[i-1])*float(K[i]))

            result_OAA[pov] += float(b)

        if(int(K[0]) == max(result_OAA, key=result_OAA.get)):
            result[K[0]] += 1
            true_classification += 1
        else:
            false_classification += 1

    print '%10s' % training_model[0].getC(),'\t', training_model[0].getTol(),'\t', result[1],'\t', result[0],'\t', result[-1],'\t', true_classification,'\t', false_classification,'\t', (float(true_classification)/float(len(testing))*100), '%'