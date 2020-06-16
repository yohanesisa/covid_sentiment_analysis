import io
import sys
import copy
import os
import math
import pandas as pd
from collections import OrderedDict
from Model.tweet import Tweet
from Model.svm import SVM
from Module.twitter import *
from Module.helper import *
# from Module.preprocessing import *
# from Module.featureExtraction import *
from Module.kernelCalculator import *
from Module.svm import *
from Module.classificationCalculator import *

class Main:

    RAW_FILE = 'Data/mentah.xlsx'

    TRAINING_FILE = 'Data/training/training.xlsx'

    TRAINING_DICT_FILE = 'Export/features/dict.xlsx'
    TRAINING_FEATURES_FILE = 'Export/features/training.xlsx'

    TESTING_FILE = 'Data/testing/testing.xlsx'
    TESTING_FEATURES_FILE = 'Export/features/testing.xlsx'

    command = ''
    while command is not 'x':
        print '-----------------------------------------------------------------------'
        print 'Command list: '
        print '    ----------   Preparation Menu   ----------'
        print '    0 - Retrieve Tweets from Twitter (output: Data/training/training.xlsx)'
        print ''
        print '    ----------   Training Menu   ----------'
        print '    1 - Preprocessing & Feature Extraction (output: Export/features/training.xlsx)'
        print '    2 - Calculate Kernel from features (output: Export/kernel/....xlsx)'
        print '    3 - Generate training model from kernel (output: Export/model/....xlsx)'
        print ''
        print '    ----------   Testing Menu   ----------'
        print '    6 - Preprocessing & Feature Extraction (output: Export/features/testing.xlsx)'
        print '    7 - Calculate Kernel from features (output: Export/kernel/testing/....xlsx)'
        print '    8 - Classication testing data'
        print ''
        print '    X - Exit Program'

        command = raw_input("Enter command: ") 

        if command is '0':
            raw = OrderedDict()
            
            for row in pd.read_excel(RAW_FILE).values.tolist():
                raw.update({ row[3].split('/')[-1].encode('utf-8'): Tweet(row[3].split('/')[-1].encode('utf-8'), '', row[4]) })

            retrieveTweets(raw)

            # Converting raw data to pandas data frame
            print('----------   Exporting Data   ----------')
            df = convertRawToDataFrame(raw)
            print df
            df.to_excel('Data/training/training.xlsx', index=False) 
            print('Exported to Data/training/training.xlsx\n')

        elif command is '1':
            training_tweets = []

            df = pd.read_excel(TRAINING_FILE)
            df = df[df['sentence'].notna()]

            for row in df.values.tolist():
                training_tweets.append(Tweet(str(row[0]),row[1],row[2]))

            # Do Preprocessing
            tweetPreprocessing(training_tweets)

            # Do Feature Extraction
            featureExtraction(training_tweets)

            # Converting training data to pandas data frame
            print('----------   Exporting Data   ----------')
            convertFeaturesToDataFrame(training_tweets).to_excel('Export/features/training.xlsx', index=False) 
            print('Exported to Export/features/training.xlsx\n')
        
        elif command is '2':
            training_features = pd.read_excel(TRAINING_FEATURES_FILE)

            sub_command = ''
            while sub_command is not 'x':
                print '-----------------------------------------------------------------------'
                print 'Command list: '
                print '    ----------   Choose Kernel   ----------'
                print '    1 - Linear'
                print '    2 - Polynomial'
                print '    3 - RBF'
                print '    4 - Sigmoid'
                print '    X - Back to main menu'

                sub_command = raw_input("Enter command : ") 

                for selected_kernel in sub_command.split(","):

                    if selected_kernel is '1':
                        linear = calculateKernelLinear(training_features)
                        linear.to_excel('Export/kernel/training/linear.xlsx', index=False)
                        print '\nExported to Export/kernel/linear.xlsx\n'

                    elif selected_kernel is '2':
                        polynomial = calculateKernelPolynomial(training_features)
                        polynomial.to_excel('Export/kernel/training/polynomial.xlsx', index=False)
                        print '\nExported to Export/kernel/training/polynomial.xlsx\n'

                    elif selected_kernel is '3':
                        gamma = raw_input("Enter gamma : ") 

                        rbf = calculateKernelRbf(training_features, gamma=float(gamma))
                        rbf.to_excel('Export/kernel/training/rbf.xlsx', index=False)
                        print '\nExported to Export/kernel/training/rbf.xlsx\n'

                    elif selected_kernel is '4':
                        a = raw_input("Enter a : ") 
                        r = raw_input("Enter r : ") 

                        sigmoid = calculateKernelSigmoid(training_features, a=float(a), r=float(r))
                        sigmoid.to_excel('Export/kernel/training/sigmoid.xlsx', index=False)
                        print '\nExported to Export/kernel/training/sigmoid.xlsx\n'

        elif command is '3':
            sub_command = ''
            while sub_command is not 'x':
                print '-----------------------------------------------------------------------'
                print 'Command list: '
                print '    ----------   Choose Kernel   ----------'
                print '    1 - Linear'
                print '    2 - Polynomial'
                print '    3 - RBF'
                print '    4 - Sigmoid'
                print '    X - Back to main menu'

                sub_command = raw_input("Enter command : ") 

                for selected_kernel in sub_command.split(","):
                    if selected_kernel is '1':
                        TRAINING_KERNEL_FILE = { 'linear':'Export/kernel/training/linear.xlsx' }
                        Cs = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
                        tols = [0.0001, 0.001, 0.01, 0.1, 1]
                        max_passes=5

                    elif selected_kernel is '2':
                        TRAINING_KERNEL_FILE = { 'polynomial':'Export/kernel/training/polynomial.xlsx' }
                        Cs = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
                        tols = [0.0001, 0.001, 0.01, 0.1, 1]
                        max_passes=5

                    elif selected_kernel is '3':
                        TRAINING_KERNEL_FILE = { 'rbf':'Export/kernel/training/rbf.xlsx' }
                        Cs = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
                        tols = [0.0001, 0.001, 0.01, 0.1, 1]
                        max_passes=5

                    elif selected_kernel is '4':
                        TRAINING_KERNEL_FILE = { 'sigmoid':'Export/kernel/training/sigmoid.xlsx' }
                        Cs = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
                        tols = [0.0001, 0.001, 0.01, 0.1, 1]
                        max_passes=5

                    else:
                        print 'Back to main menu'
                        break
                    
                    OAA = [1,0,-1]
                    training_sentiment = []

                    for kernel_type in TRAINING_KERNEL_FILE:
                        training_model = { kernel_type: [] }

                        master_kernel = pd.read_excel(TRAINING_KERNEL_FILE[kernel_type])

                        training_sentiment = master_kernel[master_kernel.columns[0]].values.tolist()
                        master_kernel = master_kernel.values.tolist()

                        print '----------   Calculate kernel ' + kernel_type + '   ----------'
                        print 'Read kernel from ' + kernel_type + '.xlsx'
                        print 'C\'s ' + str(Cs)
                        print 'Tol\'s ' + str(tols) + '\n'

                        for tol in tols:
                            for c in Cs:
                                print 'Kernel ', kernel_type,', Param SVM -> C=', c,'\ttol=', tol, '\tmax_passes=', max_passes

                                for pov in OAA:
                                    kernel = copy.deepcopy(master_kernel)
                                    for item in kernel:
                                        if item[0] == pov:                # One againts All
                                            item[0] = 1
                                        else:
                                            item[0] = -1

                                    print 'Calculating alpha bias OAA class ' + str(pov)

                                    # Search SMO
                                    result = svm(kernel, C=c, tol=tol, max_passes=max_passes)
                                    training_model[kernel_type].append(SVM(clas=pov, C=c, tol=tol, label=training_sentiment, alpha=result[0], bias=result[1]))

                        print('----------   Exporting Data   ----------')
                        convertTrainingModelToDataFrame(training_model, training_sentiment).to_excel('Export/model/'+ kernel_type +'.xlsx', index=False)
                        print('Exported to Export/model/'+ kernel_type +'.xlsx\n')

                    os.system('say "model calculation has finished"')

            
                
                

        elif command is 'xxx':
            print '----------   Load Training Model   ----------'
            print 'Read model training from Export/model/training.xlsx\n'
            model = pd.read_excel('Export/model/training.xlsx').values.tolist()

            training_model = { 'Linear': [], 'Polynomial':[], 'RBF':[], 'Sigmoid':[] }

            for index in range(1,len(model)):
                training_model[model[index][0]].append(SVM(model[index][2], model[0][4:], model[index][4:], model[index][3]))

            print '----------   Print Training Model   ----------'
            for kernel in training_model:
                print('----------   Training Model Kernel ' + kernel + '   ----------')
                for model in training_model[kernel]:
                    model.printData()
                print '\n'

        elif command is '6':
            print '----------   Load Testing Tweet   ----------'
            print 'Read tweting tweet from Data/testing/testing.xlsx\n'

            testing_tweets = []
            for tweet in pd.read_excel(TESTING_FILE).values.tolist():
                testing_tweets.append(Tweet(str(tweet[0]),tweet[1],tweet[2]))
            print 'Total tweet  : %d' % len(testing_tweets)            

            # Do Preprocessing
            tweetPreprocessing(testing_tweets)

            # Do Feature Extraction
            featureExtraction(testing_tweets, 'Testing', TRAINING_DICT_FILE)

            # Converting testing data to pandas data frame
            print('----------   Exporting Data   ----------')
            convertFeaturesToDataFrame(testing_tweets).to_excel('Export/features/testing.xlsx', index=False) 
            print('Exported to Export/features/testing.xlsx\n')

        elif command is '7':
            training_features = pd.read_excel('Export/features/training.xlsx')
            testing_features = pd.read_excel('Export/features/testing.xlsx')

            sub_command = ''
            while sub_command is not 'x':
                print '-----------------------------------------------------------------------'
                print 'Command list: '
                print '    ----------   Choose Kernel   ----------'
                print '    1 - Linear'
                print '    2 - Polynomial'
                print '    3 - RBF'
                print '    4 - Sigmoid'
                print '    X - Back to main menu'

                sub_command = raw_input("Enter command : ") 

                for selected_kernel in sub_command.split(","):
                    if selected_kernel is '1':
                        linear = calculateTestingKernelLinear(training_features, testing_features)
                        linear.to_excel('Export/kernel/testing/linear.xlsx', index=False)
                        print '\nExported to Export/kernel/testing/linear.xlsx\n'

                    elif selected_kernel is '2':
                        polynomial = calculateTestingKernelPolynomial(training_features, testing_features)
                        polynomial.to_excel('Export/kernel/testing/polynomial.xlsx', index=False)
                        print '\nExported to Export/kernel/testing/polynomial.xlsx\n'

                    elif selected_kernel is '3':
                        gamma = raw_input("Enter gamma : ") 

                        rbf = calculateTestingKernelRbf(training_features, testing_features, gamma=float(gamma))
                        rbf.to_excel('Export/kernel/testing/rbf.xlsx', index=False)
                        print '\nExported to Export/kernel/testing/rbf.xlsx\n'

                    elif selected_kernel is '4':
                        a = raw_input("Enter a : ") 
                        r = raw_input("Enter r : ") 

                        sigmoid = calculateTestingKernelSigmoid(training_features, testing_features, a=float(a), r=float(r))
                        sigmoid.to_excel('Export/kernel/testing/sigmoid.xlsx', index=False)
                        print '\nExported to Export/kernel/testing/sigmoid.xlsx\n'

        elif command is '8':
            sub_command = ''
            while sub_command is not 'x':
                print '-----------------------------------------------------------------------'
                print 'Command list: '
                print '    ----------   Choose Kernel   ----------'
                print '    1 - Linear'
                print '    2 - Polynomial'
                print '    3 - RBF'
                print '    4 - Sigmoid'
                print '    X - Back to main menu'

                sub_command = raw_input("Enter command: ") 

                if sub_command is '1':
                    print '----------   Load Training Model Kernel Linear   ----------'
                    print 'Read model training from Export/model/linear.xlsx\n'
                    model = pd.read_excel('Export/model/linear.xlsx').values.tolist()
                    kernel_type = 'linear'

                elif sub_command is '2':
                    print '----------   Load Training Model Kernel Polynomial   ----------'
                    print 'Read model training from Export/model/polynomial.xlsx\n'
                    model = pd.read_excel('Export/model/polynomial.xlsx').values.tolist()
                    kernel_type = 'polynomial'

                elif sub_command is '3':
                    print '----------   Load Training Model Kernel RBF   ----------'
                    print 'Read model training from Export/model/rbf.xlsx\n'
                    model = pd.read_excel('Export/model/rbf.xlsx').values.tolist()
                    kernel_type = 'rbf'

                elif sub_command is '4':
                    print '----------   Load Training Model Kernel Sigmoid   ----------'
                    print 'Read model training from Export/model/sigmoid.xlsx\n'
                    model = pd.read_excel('Export/model/sigmoid.xlsx').values.tolist()
                    kernel_type = 'sigmoid'
                
                else:
                    print 'Back to main menu'
                    break

                training_model = []
                index_c = 1
                while index_c < len(model)-1:
                    sub_model = []
                    for index_oaa in range(3):
                        index = index_c+index_oaa
                        sub_model.append(SVM(clas=model[index][2], C=model[index][3], tol=model[index][4], label=model[0][6:], alpha=model[index][6:], bias=model[index][5]))
                    
                    training_model.append(sub_model)
                    index_c += 3

                print '----------   Classification Model '+ kernel_type +'   ----------'
                print '%10s' % 'C','\t',"Tol",'\t',"Pos",'\t',"Net",'\t',"Neg",'\t',"True",'\t',"False",'\t',"Accuracy"
                for model_set in training_model:
                    svmClassification(training_model=model_set, kernel_type=kernel_type)
                print ''

    print 'Program terminated'

    

    
