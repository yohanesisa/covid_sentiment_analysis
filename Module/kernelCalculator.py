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

def calculateKernelRbf(data):
    print '----------   Calculating kernel RBF   ----------'

    gamma = 0.5

    data = data.values
    kernel = []
    for n, tweet in enumerate(data):                      # row
        kernel.append([data[n][0]])
        for i in range(len(data)):                        # loop over item
            sub = 0.0
            for j in range(1,len(tweet)):                 # loop over features
                sub += pow(abs(float(data[n][j])-float(data[i][j])), 2)
            
            sub = np.exp((-gamma)*(float(sub)))

            kernel[n].append(sub)

        sys.stdout.write('\r')
        sys.stdout.write("Calculating %d%%" % (float((n+1))/float(len(data))*100))
        sys.stdout.flush()

    return pd.DataFrame(kernel)

def calculateKernelSigmoid(data):
    print '----------   Calculate kernel Sigmoid   ----------'

    a = 1
    r = 1

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
        sys.stdout.write("Calculating %d%%" % (float((n+1))/float(len(data))*100))
        sys.stdout.flush()

    return pd.DataFrame(kernel)