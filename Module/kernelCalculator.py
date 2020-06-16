import pandas as pd
import numpy as np
import sys

def calculateKernelLinear(data):
    print '----------   Calculate kernel linear   ----------'
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
        sys.stdout.write("Calculating %d%%" % (float((n+1))/float(len(data))*100))
        sys.stdout.flush()

    return pd.DataFrame(kernel)

def calculateKernelPolynomial(data):
    print '----------   Calculate kernel polynomial   ----------'

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
        sys.stdout.write("Calculating %d%%" % (float((n+1))/float(len(data))*100))
        sys.stdout.flush()

    return pd.DataFrame(kernel)

def calculateKernelRbf(data, gamma):
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
        sys.stdout.write("Gamma " + str(gamma) + "\t-> Calculating %d%%" % (float((n+1))/float(len(data))*100))
        sys.stdout.flush()

    print ''

    return pd.DataFrame(kernel)

def calculateKernelSigmoid(data, a, r):
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
        sys.stdout.write("a " + str(a) + "\tr " + str(r) + "\t-> Calculating %d%%" % (float((n+1))/float(len(data))*100))
        sys.stdout.flush()

    print ''

    return pd.DataFrame(kernel)

def calculateTestingKernelLinear(training_data, testing_data):
    print '----------   Calculate kernel linear   ----------'

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
        sys.stdout.write("Calculating %d%%" % (float((u+1))/float(len(testing_data))*100))
        sys.stdout.flush()

    return pd.DataFrame(kernel)

def calculateTestingKernelPolynomial(training_data, testing_data):
    print '----------   Calculate kernel polynomial   ----------'

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
        sys.stdout.write("Calculating %d%%" % (float((u+1))/float(len(testing_data))*100))
        sys.stdout.flush()

    return pd.DataFrame(kernel)

def calculateTestingKernelRbf(training_data, testing_data, gamma):
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
        sys.stdout.write("Gamma " + str(gamma) + "\t-> Calculating %d%%" % (float((u+1))/float(len(testing_data))*100))
        sys.stdout.flush()
    
    print ''

    return pd.DataFrame(kernel)

def calculateTestingKernelSigmoid(training_data, testing_data, a, r):
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
        sys.stdout.write("a " + str(a) + "\tr " + str(r) + "\t-> Calculating %d%%" % (float((u+1))/float(len(testing_data))*100))
        sys.stdout.flush()
    
    print ''

    return pd.DataFrame(kernel)