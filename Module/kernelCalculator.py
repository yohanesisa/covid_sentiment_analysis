import pandas as pd
import numpy as np
import sys
from Export.parameter import *

def calculateKernelLinear(data):
    data = data.values
    kernel = []
    for n, tweet in enumerate(data):                      # row
        kernel.append([data[n][0]])
        for i in range(len(data)):                        # loop over item
            sub = 0.0
            for j in range(1,len(tweet)):                  # loop over features
                sub += (float(data[n][j])*float(data[i][j]))
            
            kernel[n].append(sub)

        sys.stdout.write('\r')
        sys.stdout.write('Kernel %s -> Calculating %d%%' % (str(1)+' from '+str(1),float((n+1))/float(len(data))*100))
        sys.stdout.flush()

    print ''

    return pd.DataFrame(kernel)

def calculateKernelPolynomial(data):
    data = data.values
    kernel = []
    
    for n, tweet in enumerate(data):                      # row
        kernel.append([data[n][0]])
        for i in range(len(data)):                        # loop over item
            sub = 0.0
            for j in range(1,len(tweet)):                 # loop over features
                sub += float(data[n][j])*float(data[i][j])
            
            sub = pow(1 + float(sub), 2)

            kernel[n].append(sub)

        sys.stdout.write('\r')
        sys.stdout.write('Kernel %s -> Calculating %d%%' % (str(1)+' from '+str(1),float((n+1))/float(len(data))*100))
        sys.stdout.flush()
    
    print ''

    return pd.DataFrame(kernel)

def calculateKernelRbf(data, index_gamma, gamma):
    data = data.values
    kernel = []
    
    for n, tweet in enumerate(data):                      # row
        kernel.append([data[n][0]])
        for i in range(len(data)):                        # loop over item
            sub = 0.0
            for j in range(1,len(tweet)):                 # loop over features
                sub += pow(float(data[n][j])-float(data[i][j]), 2)
            
            sub = np.exp((-gamma)*(float(sub)))

            kernel[n].append(sub)

        sys.stdout.write('\r')
        sys.stdout.write('Kernel %s -> Calculating %d%%' % (str(index_gamma+1)+' from '+str(len(gammas)),float((n+1))/float(len(data))*100))
        sys.stdout.flush()

    print ''

    return pd.DataFrame(kernel)

def calculateKernelSigmoid(data, index_a, index_r, a, r):
    data = data.values
    kernel = []

    for n, tweet in enumerate(data):                      # row
        kernel.append([data[n][0]])
        for i in range(len(data)):                        # loop over item
            sub = 0.0
            for j in range(1,len(tweet)):                 # loop over features
                sub += float(data[n][j])*float(data[i][j])
            
            sub = np.tanh(float(a)*float(sub)+float(r))

            kernel[n].append(sub)

        sys.stdout.write('\r')
        sys.stdout.write('Kernel %s -> Calculating %d%%' % (str((len(aa)*index_r)+(index_a+1))+' from '+str(len(aa)*len(rr)),float((n+1))/float(len(data))*100))
        sys.stdout.flush()

    print ''

    return pd.DataFrame(kernel)

def calculateTestingKernelLinear(training_data, testing_data):
    training_data = training_data.values
    testing_data = testing_data.values
    kernel = []

    for u, testing in enumerate(testing_data):
        kernel.append([testing_data[u][0]])
        for t in range(len(training_data)):
            sub = 0.0
            for j in range(1,len(testing)):
                sub += (float(testing_data[u][j])*float(training_data[t][j]))
            
            kernel[u].append(sub)

        sys.stdout.write('\r')
        sys.stdout.write('Kernel %s -> Calculating %d%%' % (str(1)+' from '+str(1),float((u+1))/float(len(testing_data))*100))
        sys.stdout.flush()

    print ''

    return pd.DataFrame(kernel)

def calculateTestingKernelPolynomial(training_data, testing_data):
    training_data = training_data.values
    testing_data = testing_data.values
    kernel = []

    for u, testing in enumerate(testing_data):
        kernel.append([testing_data[u][0]])
        for t in range(len(training_data)):
            sub = 0.0
            for j in range(1,len(testing)):
                sub += float(testing_data[u][j])*float(training_data[t][j])
            
            sub = pow(1 + float(sub), 2)

            kernel[u].append(sub)

        sys.stdout.write('\r')
        sys.stdout.write('Kernel %s -> Calculating %d%%' % (str(1)+' from '+str(1),float((u+1))/float(len(testing_data))*100))
        sys.stdout.flush()

    print ''

    return pd.DataFrame(kernel)

def calculateTestingKernelRbf(training_data, testing_data, index_gamma, gamma):
    training_data = training_data.values
    testing_data = testing_data.values
    kernel = []

    for u, testing in enumerate(testing_data):
        kernel.append([testing_data[u][0]])
        for t in range(len(training_data)):
            sub = 0.0
            for j in range(1,len(testing)):
                sub += pow(float(testing_data[u][j])-float(training_data[t][j]), 2)
            
            sub = np.exp((-gamma)*(float(sub)))

            kernel[u].append(sub)

        sys.stdout.write('\r')
        sys.stdout.write('Kernel %s -> Calculating %d%%' % (str(index_gamma+1)+' from '+str(len(gammas)),float((u+1))/float(len(testing_data))*100))
        sys.stdout.flush()
    
    print ''

    return pd.DataFrame(kernel)

def calculateTestingKernelSigmoid(training_data, testing_data, index_a, a, index_r, r):
    training_data = training_data.values
    testing_data = testing_data.values
    kernel = []
    
    for u, testing in enumerate(testing_data):
        kernel.append([testing_data[u][0]])
        for t in range(len(training_data)):
            sub = 0.0
            for j in range(1,len(testing)):
                sub += float(testing_data[u][j])*float(training_data[t][j])
            
            sub = np.tanh(float(a)*float(sub)+float(r))

            kernel[u].append(sub)

        sys.stdout.write('\r')
        sys.stdout.write('Kernel %s -> Calculating %d%%' % (str((len(aa)*index_r)+(index_a+1))+' from '+str(len(aa)*len(rr)),float((u+1))/float(len(testing_data))*100))
        sys.stdout.flush()
    
    print ''

    return pd.DataFrame(kernel)