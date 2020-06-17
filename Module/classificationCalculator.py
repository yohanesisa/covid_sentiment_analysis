import copy
import pandas as pd
from Model.result import Result

def svmClassification(training_model, testing_kernel, kernel_type):
    result = { 1: 0, 0: 0, -1: 0 }
    true_classification = 0
    false_classification = 0

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

        if(int(K[0]) == max(result_OAA, key=result_OAA.get)):
            result[K[0]] += 1
            true_classification += 1
        else:
            false_classification += 1

    accuracy = (float(true_classification)/float(len(testing_kernel))*100)

    return Result(kernel=kernel_type, C=training_model[0].getC(), tol=training_model[0].getTol(), gamma=training_model[0].getGamma(), a=training_model[0].getA(), r=training_model[0].getR(), accuracy=accuracy)
