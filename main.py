import io
import sys
import copy
import os
import math
import time
import pandas as pd
from collections import OrderedDict
from Model.tweet import Tweet
from Model.svm import SVM
from Module.twitter import *
from Module.helper import *
from Module.preprocessing import *
# from Module.featureExtraction import *
# from Module.kernelCalculator import *
from Module.svm import *
from Module.classificationCalculator import *
from Export.parameter import *

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
                        linear.to_excel('Export/kernel/training/linear.xlsx', sheet_name=str('linear'), index=False)
                        print '\nExported to Export/kernel/linear.xlsx\n'

                    elif selected_kernel is '2':
                        polynomial = calculateKernelPolynomial(training_features)
                        polynomial.to_excel('Export/kernel/training/polynomial.xlsx', sheet_name=str('polynomial'), index=False)
                        print '\nExported to Export/kernel/training/polynomial.xlsx\n'

                    elif selected_kernel is '3':
                        print '----------   Calculating kernel RBF   ----------'

                        print "Gamma's ", str(gammas)

                        writer = pd.ExcelWriter('Export/kernel/training/rbf.xlsx')

                        for index_gamma, gamma in enumerate(gammas):
                            rbf = calculateKernelRbf(training_features, index_gamma=index_gamma, gamma=float(gamma))
                            rbf.to_excel(writer, sheet_name=str(gamma), index=False)

                        writer.save()
                        print '\nExported to Export/kernel/training/rbf.xlsx\n'

                    elif selected_kernel is '4':
                        print '----------   Calculating kernel Sigmoid   ----------'

                        print "a's ", str(aa)
                        print "r's ", str(rr)

                        writer = pd.ExcelWriter('Export/kernel/training/sigmoid.xlsx')

                        for index_r, r in enumerate(rr):
                            for index_a, a in enumerate(aa):
                                sigmoid = calculateKernelSigmoid(training_features, index_a=index_a, index_r=index_r, a=float(a), r=float(r))
                                sigmoid.to_excel(writer, sheet_name=str(a)+';'+str(r), index=False)

                        writer.save()
                        print '\nExported to Export/kernel/training/sigmoid.xlsx\n'

                os.system('say "training kernel generated"')  

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
                        total_model = total_model_linear

                    elif selected_kernel is '2':
                        TRAINING_KERNEL_FILE = { 'polynomial':'Export/kernel/training/polynomial.xlsx' }
                        total_model = total_model_polynomial

                    elif selected_kernel is '3':
                        TRAINING_KERNEL_FILE = { 'rbf':'Export/kernel/training/rbf.xlsx' }
                        total_model = total_model_rbf

                    elif selected_kernel is '4':
                        TRAINING_KERNEL_FILE = { 'sigmoid':'Export/kernel/training/sigmoid.xlsx' }
                        total_model = total_model_sigmoid

                    else:
                        print 'Back to main menu'
                        break
                    
                    OAA = [1,0,-1]
                    training_sentiment = []

                    for kernel_type in TRAINING_KERNEL_FILE:
                        training_model = { kernel_type: [] }

                        kernels = pd.read_excel(TRAINING_KERNEL_FILE[kernel_type], sheet_name=None)

                        counter_model = 0

                        for sheet in kernels:
                            sheetDF = pd.DataFrame(kernels[sheet])
                            training_sentiment = sheetDF[sheetDF.columns[0]].values.tolist()

                            master_kernel = sheetDF.values.tolist()

                            print '\n----------   Calculate kernel ' + kernel_type + '   ----------'
                            print 'Read kernel from ' + kernel_type + '.xlsx'
                            print '\nC\'s \t' + str(Cs)
                            print 'Tol\'s \t' + str(tols)

                            if kernel_type == 'rbf':
                                print 'Gamma \t' + str(sheet)
                            
                            if kernel_type == 'sigmoid':
                                print 'a \t' + str(sheet.split(';')[0])
                                print 'r \t' + str(sheet.split(';')[1])

                            print ''

                            for tol in tols:
                                for c in Cs:
                                    for pov in OAA:
                                        kernel = copy.deepcopy(master_kernel)
                                        for item in kernel:
                                            if item[0] == pov:                # One againts All
                                                item[0] = 1
                                            else:
                                                item[0] = -1

                                        sys.stdout.write('\r')
                                        sys.stdout.write('Model %s -> Calculating...' % (str(counter_model+1)+' from '+str(total_model)))
                                        sys.stdout.flush()

                                        # Search SMO
                                        result = svm(pov=pov, K=kernel, C=c, tol=tol, max_passes=max_passes)

                                        counter_model += 1

                                        if kernel_type == 'rbf':
                                            training_model[kernel_type].append(SVM(clas=pov, C=c, tol=tol, gamma=sheet, a='-', r='-', label=training_sentiment, alpha=result[0], bias=result[1]))
                                        elif kernel_type == 'sigmoid':
                                            training_model[kernel_type].append(SVM(clas=pov, C=c, tol=tol, gamma='-', a=sheet.split(';')[0], r=sheet.split(';')[1], label=training_sentiment, alpha=result[0], bias=result[1]))
                                        else:
                                            training_model[kernel_type].append(SVM(clas=pov, C=c, tol=tol, gamma='-', a='-', r='-', label=training_sentiment, alpha=result[0], bias=result[1]))

                        print('\n----------   Exporting Data   ----------')
                        convertTrainingModelToDataFrame(training_model, training_sentiment).to_excel('Export/model/'+ kernel_type +'.xlsx', index=False)
                        print('Exported to Export/model/'+ kernel_type +'.xlsx\n')

                        os.system('say "model '+ kernel_type +' generated"')     

        # ------------------------------------------------------------------------------------------------------------------------------------------------------

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
                        linear.to_excel('Export/kernel/testing/linear.xlsx', sheet_name=str('linear'), index=False)
                        print '\nExported to Export/kernel/testing/linear.xlsx\n'

                    elif selected_kernel is '2':
                        polynomial = calculateTestingKernelPolynomial(training_features, testing_features)
                        polynomial.to_excel('Export/kernel/testing/polynomial.xlsx', sheet_name=str('polynomial'), index=False)
                        print '\nExported to Export/kernel/testing/polynomial.xlsx\n'

                    elif selected_kernel is '3':
                        print '----------   Calculating kernel RBF   ----------'

                        print "Gamma's ", str(gammas)

                        writer = pd.ExcelWriter('Export/kernel/testing/rbf.xlsx')

                        for index_gamma, gamma in enumerate(gammas):
                            rbf = calculateTestingKernelRbf(training_features, testing_features, index_gamma=index_gamma, gamma=float(gamma))
                            rbf.to_excel(writer, sheet_name=str(gamma), index=False)

                        writer.save()
                        print '\nExported to Export/kernel/testing/rbf.xlsx\n'

                    elif selected_kernel is '4':
                        print '----------   Calculating kernel Sigmoid   ----------'

                        print "a's ", str(aa)
                        print "r's ", str(rr)

                        writer = pd.ExcelWriter('Export/kernel/testing/sigmoid.xlsx')

                        for index_r, r in enumerate(rr):
                            for index_a, a in enumerate(aa):
                                sigmoid = calculateTestingKernelSigmoid(training_features, testing_features, index_a=index_a, index_r=index_r, a=float(a), r=float(r))
                                sigmoid.to_excel(writer, sheet_name=str(a)+';'+str(r), index=False)

                        writer.save()
                        print '\nExported to Export/kernel/testing/sigmoid.xlsx\n'

                os.system('say "testing kernel generated"')  

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

                timeObj = time.localtime(time.time())
                timestamp = '%d-%d-%d %d:%d:%d' % (timeObj.tm_mday, timeObj.tm_mon, timeObj.tm_year, timeObj.tm_hour, timeObj.tm_min, timeObj.tm_sec)

                for selected_kernel in sub_command.split(","):
                    if selected_kernel is '1':
                        print '----------   Load Training Model Kernel Linear   ----------'
                        print 'Read model training from Export/model/linear.xlsx'
                        model = pd.read_excel('Export/model/linear.xlsx').values.tolist()
                        kernel_type = 'linear'

                    elif selected_kernel is '2':
                        print '----------   Load Training Model Kernel Polynomial   ----------'
                        print 'Read model training from Export/model/polynomial.xlsx'
                        model = pd.read_excel('Export/model/polynomial.xlsx').values.tolist()
                        kernel_type = 'polynomial'

                    elif selected_kernel is '3':
                        print '----------   Load Training Model Kernel RBF   ----------'
                        print 'Read model training from Export/model/rbf.xlsx'
                        model = pd.read_excel('Export/model/rbf.xlsx').values.tolist()
                        kernel_type = 'rbf'

                    elif selected_kernel is '4':
                        print '----------   Load Training Model Kernel Sigmoid   ----------'
                        print 'Read model training from Export/model/sigmoid.xlsx'
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
                            sub_model.append(SVM(clas=model[index][2], C=model[index][3], tol=model[index][4], gamma=model[index][5], a=model[index][6], r=model[index][7], label=model[0][9:], alpha=model[index][9:], bias=model[index][8]))
                        training_model.append(sub_model)
                        index_c += 3

                    print 'Read testing kernel from Export/kernel/testing/'+kernel_type+'.xlsx\n'
                    if kernel_type == 'linear':
                        master_testing_kernel = pd.read_excel('Export/kernel/testing/linear.xlsx', sheet_name='linear').values.tolist()
                    elif kernel_type == 'polynomial':
                        master_testing_kernel = pd.read_excel('Export/kernel/testing/polynomial.xlsx', sheet_name='polynomial').values.tolist()
                    elif kernel_type == 'rbf':
                        master_testing_kernel = pd.read_excel('Export/kernel/testing/rbf.xlsx', sheet_name=None)
                    elif kernel_type == 'sigmoid':
                        master_testing_kernel = pd.read_excel('Export/kernel/testing/sigmoid.xlsx', sheet_name=None)

                    result_training_model = []
                    for index_model_set, model_set in enumerate(training_model):

                        # sys.stdout.write('\r')
                        # sys.stdout.write("Calculating %d%%" % (float((index_model_set+1))/float(len(training_model))*100))
                        # sys.stdout.flush()

                        if kernel_type == 'rbf':
                            testing_kernel = master_testing_kernel[model_set[0].getGamma()].values
                        elif kernel_type == 'sigmoid':
                            testing_kernel = master_testing_kernel[model_set[0].getA()+';'+model_set[0].getR()].values
                        else:
                            testing_kernel = master_testing_kernel

                        result_training_model.append(svmClassification(training_model=model_set, testing_kernel=testing_kernel, kernel_type=kernel_type))
                    
                    print('----------   Exporting Data   ----------')
                    convertResultToDataFrame(result_training_model).to_excel('Export/result/'+timestamp+' '+kernel_type+'.xlsx', index=False) 
                    print('Exported to Export/result/'+timestamp+' '+kernel_type+'.xlsx\n')

                    os.system('say "classification '+kernel_type+' has finished"')  
                
        elif command is 'p':
            print 'C          : ', len(Cs), ' ', str(Cs)
            print 'tols       : ', len(tols), ' ', str(tols)
            print 'max_passes : ', str(max_passes)
            print ''
            print 'gamma      : ', len(gammas), ' ', str(gammas)
            print ''
            print 'a          : ', len(aa), ' ', str(aa)
            print 'r          : ', len(rr), ' ', str(rr)

    print 'Program terminated'

    

    
