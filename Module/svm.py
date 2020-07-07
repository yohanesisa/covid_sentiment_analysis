import random
import sys
import numpy as np
from Module.helper import *

def svm(pov, K, C, tol):    #SMO function
    a = [0.0] * len(K)
    a_old = [0.0] * len(K)
    b = 0.0
    passes = 0
    var = 0.00001
    max_passes = 1

    y = []
    for row in K:
        y.append(int(row[0]))
        del row[0]

    E = [0.0] * len(K)
    while passes < max_passes:

        # sys.stdout.write('\r')
        # sys.stdout.write("POV %2d -> C: %10s Tol: %5s -> Pass %s: Calculating..." % (pov, str(C), str(tol), passes+1))
        # sys.stdout.flush()

        num_changed_alphas = 0
        for i in range(len(K)):
            E[i] = calculateE(K[i], y, a, b, i)

            if ((float(y[i])*float(E[i]) < float(-tol)) and (float(a[i]) < float(C))) or ((float(y[i])*float(E[i]) > float(tol)) and (float(a[i]) > float(C))):
                j = randomJ(i, len(K))
                E[j] = calculateE(K[j], y, a, b, j)

                a_old[i] = a[i]
                a_old[j] = a[j]

                #Compute L & H
                if y[i] != y[j]:
                    L = max(0.0,float(a[j])-float(a[i]))
                    H = min(float(C),float(C)+float(a[j])-float(a[i]))
                else:
                    L = max(0.0,float(a[i])+float(a[j])-float(C))
                    H = min(float(C),float(a[i])+float(a[j]))

                if L == H:
                    continue
                
                eta = float(2)*float(K[i][j]) - float(K[i][i]) - float(K[j][j])
                if eta >= 0.0:
                    continue

                candidate_a = a[j] - ((float(y[j]*(float(E[i])-float(E[j]))))/float(eta))
                if candidate_a > H:
                    a[j] = H
                elif candidate_a < L:
                    a[j] = L
                else:
                    a[j] = candidate_a

                if abs(float(a[j])-float(a_old[j]) < var):
                    continue
                
                a[i] = float(a[i])+float(y[i])*float(y[j])*(float(a_old[j])-float(a[j]))

                b1 = float(b) - float(E[i]) - float(y[i])*(float(a[i]-a_old[i]))*(float(K[i][i])) - float(y[j])*(float(a[j]-a_old[j]))*(float(K[i][j]))
                b2 = float(b) - float(E[j]) - float(y[i])*(float(a[i]-a_old[i]))*(float(K[i][j])) - float(y[j])*(float(a[j]-a_old[j]))*(float(K[j][j]))

                if float(0) < float(a[i]) and float(a[i]) < float(C):
                    b = b1
                elif float(0) < float(a[j]) and float(a[j]) < float(C):
                    b = b2
                else:
                    b = (float(b1)+float(b2))/float(2)

                num_changed_alphas += 1

        if num_changed_alphas == 0:
            passes += 1
        else:
            passes = 0
    
    # print '\n'

    return a, b

def calculateE(K, y, a, b, i):
    E = 0.0
    for m in range(len(K)):
        E += (float(a[m])*float(y[m])*float(K[m]))

    E += float(b)
    E -= float(y[i])

    return E

def randomJ(i, length):
    j = i
    while j == i:
        j = random.randint(0,length-1)
    
    return j