import io
import sys
import copy
import pandas as pd
from collections import OrderedDict
from Model.tweet import Tweet
from Model.svm import SVM
from Module.helper import *
# from Module.preprocessing import *
# from Module.featureExtraction import *
from Module.kernelCalculator import *
from Module.svm import *

class Main:

    # trainingCSV = [i.strip().split(';') for i in io.open('data/training.csv', encoding='utf-8-sig')]
    training_csv = [i.strip().split(';') for i in io.open('data/trainingBeda.csv', encoding='utf-8-sig')]
    # trainingCSV = [i.strip().split(';') for i in io.open('data/trainingDummy.csv', encoding='utf-8-sig')]

    # training_features = pd.read_excel('Export/features.xlsx')
    training_features = pd.read_excel('Export/features-beda.xlsx')
    # training_features = pd.read_excel('Export/features-analisis-manual.xlsx')

    training_kernel = { 'Polynomial':'Export/kernel/polynomial.xlsx' }
    # training_kernel = { 'Linear':'Export/kernel/linear.xlsx', 'Polynomial':'Export/kernel/polynomial.xlsx', 'RBF':'Export/kernel/rbf.xlsx', 'Sigmoid':'Export/kernel/sigmoid.xlsx' }
    # training_kernel = { 'Linear':'Export/kernel9/linear.xlsx', 'Polynomial':'Export/kernel9/polynomial.xlsx', 'RBF':'Export/kernel9/rbf.xlsx', 'Sigmoid':'Export/kernel9/sigmoid.xlsx' }
    
    training_model = { 'Linear': [], 'Polynomial':[], 'RBF':[], 'Sigmoid':[] }

    # Load data from dataset
    training_tweets = []
    for index, row in enumerate(training_csv):
        training_tweets.append(Tweet(index,row[0].encode('utf-8'),row[1].encode('utf-8')))
    print('----------   Load Data   ----------')
    print('Total tweet  : %d' % len(training_tweets))
    print('Positive : %d' % countSent(training_tweets, 'positive'))
    print('Negative : %d' % countSent(training_tweets, 'negative'))
    print('Neutral  : %d' % countSent(training_tweets, 'neutral'))

    command = ''
    while command is not 'x':
        print '-----------------------------------------------------------------------'
        print 'Command list: '
        print '    1 - Preprocessing & Feature Extraction (output: features.xlsx)'
        print '    2 - Calculate Kernel from features'
        print '    3 - Generate training model from kernel'
        print '    4 - Print training model'
        print '    X - Exit Program'

        command = raw_input("Enter command: ") 

        if command is '1':
            # Do Preprocessing
            tweetPreprocessing(training_tweets)

            # Do Feature Extraction
            featureExtraction(training_tweets)

            # Converting training data to pandas data frame
            print('----------   Exporting Data   ----------')
            trainingDF = convertToDataFrame(training_tweets)
            trainingDF.to_excel('Export/features.xlsx', index=False) 
            print('Exported to Export/features.xlsx\n')
        
        elif command is '2':
            linear = calculateKernelLinear(training_features)
            linear.to_excel('Export/kernel/linear.xlsx', index=False)
            print '\nExported to Export/kernel/linear.xlsx\n'

            polynomial = calculateKernelPolynomial(training_features)
            polynomial.to_excel('Export/kernel/polynomial.xlsx', index=False)
            print '\nExported to Export/kernel/polynomial.xlsx\n'

            rbf = calculateKernelRbf(training_features)
            rbf.to_excel('Export/kernel/rbf.xlsx', index=False)
            print '\nExported to Export/kernel/rbf.xlsx\n'

            sigmoid = calculateKernelSigmoid(training_features)
            sigmoid.to_excel('Export/kernel/sigmoid.xlsx', index=False)
            print '\nExported to Export/kernel/sigmoid.xlsx\n'

        elif command is '3':
            sentimentLabel = [1,-1,0]

            for excel in training_kernel:
                master_kernel = pd.read_excel(training_kernel[excel]).values.tolist()
                print '----------   Calculate kernel ' + excel + '   ----------'
                print 'Read kernel from ' + excel + '.xlsx\n'

                for label in sentimentLabel:
                    kernel = copy.deepcopy(master_kernel)
                    for item in kernel:
                        if item[0] == label:                # One againts All
                            item[0] = 1
                        else:
                            item[0] = -1

                    print 'Calculating alpha bias OAA class ' + str(label)

                    # Search SMO
                    temp = smo(kernel, 10, 0.001, 5)
                    training_model[excel].append(SVM(label, temp[0], temp[1]))
        
        elif command is '4':
            for excel in training_kernel:
                print '----------   Training model kernel ' + excel + '   ----------'
                for item in training_model[excel]:
                    item.printData()
            print '\n'

    
    print 'Program terminated'

    

    
