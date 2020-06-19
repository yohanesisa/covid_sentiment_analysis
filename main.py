import sys
import copy
import os
import time
import pandas as pd
import numpy as np
from collections import OrderedDict
from Model.tweet import Tweet
from Model.svm import SVM
from Module.twitter import *
from Module.helper import *
from Module.preprocessing import *
from Module.featureExtraction import *
from Module.kernelCalculator import *
from Module.svm import *
from Module.classificationCalculator import *
from Export.parameter import *

class Main:

    RAW_FILE = 'Data/mentah.xlsx'

    TRAINING_FILE = 'Data/training/training.xlsx'

    TRAINING_DICT_FILE = 'Export/features/dict.xlsx'
    TRAINING_FEATURES_FILE = 'Export/features/training.xlsx'

    TESTING_FILE = 'Data/testing/testing.xlsx'

    command = ''
    while command is not 'x':
        print '-----------------------------------------------------------------------'
        print 'Command list: '
        print '    ----------   Dataset Menu   ----------'
        print '    a - Retrieve Tweets from Twitter'
        print '    b - Split Data Tweets to Training & Testing'
        print ''
        print '    ----------   Training Menu   ----------'
        print '    1 - Training Preprocessing & Feature Extraction'
        print '    2 - Calculate Training Kernel from Features'
        print '    3 - Generate Training Model from Kernel'
        print ''
        print '    ----------   Testing Menu   ----------'
        print '    6 - Testing Preprocessing & Feature Extraction'
        print '    7 - Calculate Testing Kernel from Features'
        print '    8 - Classication Testing Data'
        print ''
        print '    p - Print Hyperparameter'
        print '    x - Exit Program'

        command = raw_input("Enter command: ") 

        if command is 'a':
            print '\n----------   Retrieve Tweets from Twitter   ----------'
            print 'Read data tweet from', RAW_FILE
            raw = OrderedDict()
            
            for row in pd.read_excel(RAW_FILE).values.tolist():
                raw.update({ row[3].split('/')[-1].encode('utf-8'): Tweet(row[3].split('/')[-1].encode('utf-8'), '', row[5]) }) #use 6 person label

            retrieveTweets(raw)

            # Converting raw data to pandas data frame
            print 'Collected',len(raw),'tweets from twitter.com'
            convertRawToDataFrame(raw).to_excel('Data/tweets.xlsx', index=False) 
            print 'Exported to Data/tweets.xlsx\n'

        elif command is 'b':
            print '\n----------   Split Data Tweets to Training & Testing   ----------'
            print 'Read data tweet from Data/tweets.xlsx'

            tweets = { 1:[], 0:[], -1:[] }
            empty = 0

            for tweet in pd.read_excel('Data/tweets.xlsx').replace(np.nan, '', regex=True).values.tolist():
                if(tweet[1] == ''):
                    empty += 1
                else:
                    if(tweet[2] == 1):
                        tweets[1].append(Tweet(str(tweet[0]),tweet[1],tweet[2]))
                    elif(tweet[2] == 0):
                        tweets[0].append(Tweet(str(tweet[0]),tweet[1],tweet[2]))
                    elif(tweet[2] == -1):
                        tweets[-1].append(Tweet(str(tweet[0]),tweet[1],tweet[2]))

            print '\nTotal pos   : ', len(tweets[1])
            print 'Total net   : ', len(tweets[0])
            print 'Total neg   : ', len(tweets[-1])
            print 'Total empty : ', empty

            random.shuffle(tweets[1])
            random.shuffle(tweets[0])
            random.shuffle(tweets[-1])

            picked_tweets = { 1:[], 0:[], -1:[] }
            min_len = min(len(tweets[1]), len(tweets[0]), len(tweets[-1]))

            picked_tweets = { 1:[], 0:[], -1:[] }
            picked_tweets[1].extend(tweets[1][0:min_len])
            picked_tweets[0].extend(tweets[0][0:min_len])
            picked_tweets[-1].extend(tweets[-1][0:min_len])

            del tweets[1][0:min_len]
            del tweets[0][0:min_len]
            del tweets[-1][0:min_len]

            training_tweets = []
            testing_tweets  = []
            leftover_tweets = []

            ratio = int(np.ceil(float(2)/float(3)*min_len))

            training_tweets.extend(picked_tweets[1][0:ratio])
            training_tweets.extend(picked_tweets[0][0:ratio])
            training_tweets.extend(picked_tweets[-1][0:ratio])

            testing_tweets.extend(picked_tweets[1][ratio:])
            testing_tweets.extend(picked_tweets[0][ratio:])
            testing_tweets.extend(picked_tweets[-1][ratio:])

            leftover_tweets.extend(tweets[1])
            leftover_tweets.extend(tweets[0])
            leftover_tweets.extend(tweets[-1])

            # Converting raw data to pandas data frame
            convertTweetsToDataFrame(training_tweets).to_excel('Data/training/training.xlsx', index=False) 
            print '\nExported to Data/training/training.xlsx'
            convertTweetsToDataFrame(testing_tweets).to_excel('Data/testing/testing.xlsx', index=False) 
            print 'Exported to Data/testing/testing.xlsx'
            convertTweetsToDataFrame(leftover_tweets).to_excel('Data/leftover.xlsx', index=False) 
            print 'Exported to Data/leftover.xlsx\n'      

        elif command is '1':
            print '\n----------   Training Preprocessing & Feature Extraction   ----------'
            print 'Read data tweet from', TRAINING_FILE

            training_tweets = []
            for row in pd.read_excel(TRAINING_FILE).values.tolist():
                training_tweets.append(Tweet(str(row[0]),row[1],row[2]))

            # Do Preprocessing
            tweetPreprocessing(training_tweets)

            # Do Feature Extraction
            featureExtraction(training_tweets)

            # Converting training data to pandas data frame
            convertFeaturesToDataFrame(training_tweets).to_excel('Export/features/training.xlsx', index=False) 
            print 'Exported to Export/features/training.xlsx\n'
        
        elif command is '2':
            training_features = pd.read_excel(TRAINING_FEATURES_FILE)

            sub_command = ''
            while sub_command is not 'x':
                print '-----------------------------------------------------------------------'
                print 'Choose Kernel: '
                print '    ----------   Calculate Training Kernel   ----------'
                print '    1 - Linear'
                print '    2 - Polynomial'
                print '    3 - RBF'
                print '    4 - Sigmoid'
                print '    x - Back to main menu'

                sub_command = raw_input("Enter command : ")
                for selected_kernel in sub_command.split(","):
                    if selected_kernel is '1':
                        print '\n----------   Calculating Training Kernel Linear   ----------'

                        linear = calculateKernelLinear(training_features)
                        linear.to_excel('Export/kernel/training/linear.xlsx', sheet_name=str('linear'), index=False)
                        print 'Exported to Export/kernel/linear.xlsx'
                        
                        os.system('say "training kernel linear generated"')

                    elif selected_kernel is '2':
                        print '\n----------   Calculating  Training Kernel Polynomial   ----------'

                        polynomial = calculateKernelPolynomial(training_features)
                        polynomial.to_excel('Export/kernel/training/polynomial.xlsx', sheet_name=str('polynomial'), index=False)
                        print 'Exported to Export/kernel/training/polynomial.xlsx'

                        os.system('say "training kernel polynomial generated"')

                    elif selected_kernel is '3':
                        print '\n----------   Calculating Training Kernel RBF   ----------'

                        print "Gamma's ", str(gammas)
                        print ''
                        writer = pd.ExcelWriter('Export/kernel/training/rbf.xlsx')
                        for index_gamma, gamma in enumerate(gammas):
                            rbf = calculateKernelRbf(training_features, index_gamma=index_gamma, gamma=float(gamma))
                            rbf.to_excel(writer, sheet_name=str(gamma), index=False)
                        writer.save()
                        print 'Exported to Export/kernel/training/rbf.xlsx'

                        os.system('say "training kernel rbf generated"')

                    elif selected_kernel is '4':
                        print '\n----------   Calculating  Training Kernel Sigmoid   ----------'

                        print "a's ", str(aa)
                        print "r's ", str(rr)
                        print ''
                        writer = pd.ExcelWriter('Export/kernel/training/sigmoid.xlsx')
                        for index_r, r in enumerate(rr):
                            for index_a, a in enumerate(aa):
                                sigmoid = calculateKernelSigmoid(training_features, index_a=index_a, index_r=index_r, a=float(a), r=float(r))
                                sigmoid.to_excel(writer, sheet_name=str(a)+';'+str(r), index=False)
                        writer.save()
                        print 'Exported to Export/kernel/training/sigmoid.xlsx'

                        os.system('say "training kernel sigmoid generated"')

                    else:
                        print 'Back to main menu'
                        break

        elif command is '3':
            sub_command = ''
            while sub_command is not 'x':
                print '-----------------------------------------------------------------------'
                print 'Choose Kernel: '
                print '    ----------   Generate Training Model from Kernel   ----------'
                print '    1 - Linear'
                print '    2 - Polynomial'
                print '    3 - RBF'
                print '    4 - Sigmoid'
                print '    x - Back to main menu'

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
                        timeObj = time.localtime(time.time())
                        print '\n----------   Calculate model kernel ' + kernel_type + '   ----------'
                        print 'Started at %d-%d-%d %d:%d:%d' % (timeObj.tm_mday, timeObj.tm_mon, timeObj.tm_year, timeObj.tm_hour, timeObj.tm_min, timeObj.tm_sec)
                        print 'Read kernel from ' + kernel_type + '.xlsx'

                        training_model = { kernel_type: [] }

                        kernels = pd.read_excel(TRAINING_KERNEL_FILE[kernel_type], sheet_name=None)

                        counter_model = 0

                        for sheet in kernels:
                            sheetDF = pd.DataFrame(kernels[sheet])
                            training_sentiment = sheetDF[sheetDF.columns[0]].values.tolist()

                            master_kernel = sheetDF.values.tolist()

                            print '\nC\'s \t' + str(Cs)
                            print 'Tol\'s \t' + str(tols)

                            if kernel_type == 'rbf':
                                print 'Gamma \t' + str(sheet)
                            
                            if kernel_type == 'sigmoid':
                                print 'a \t' + str(sheet.split(';')[0])
                                print 'r \t' + str(sheet.split(';')[1])

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
                                        sys.stdout.write('Model \t%s -> Calculating...' % (str(counter_model+1)+' from '+str(total_model)))
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
                            
                            print ''
                        
                        timeObj = time.localtime(time.time())
                        print '\nFinished at %d-%d-%d %d:%d:%d' % (timeObj.tm_mday, timeObj.tm_mon, timeObj.tm_year, timeObj.tm_hour, timeObj.tm_min, timeObj.tm_sec)

                        convertTrainingModelToDataFrame(training_model, training_sentiment).to_excel('Export/model/'+ kernel_type +'.xlsx', index=False)
                        print 'Exported to Export/model/'+ kernel_type +'.xlsx'

                        os.system('say "model '+ kernel_type +' generated"')     

        # ------------------------------------------------------------------------------------------------------------------------------------------------------

        elif command is '6':
            print '\n----------   Load Testing Tweet   ----------'
            print 'Read data tweet from', TESTING_FILE

            testing_tweets = []
            for tweet in pd.read_excel(TESTING_FILE).values.tolist():
                testing_tweets.append(Tweet(str(tweet[0]),tweet[1],tweet[2]))
            print 'Total Tweet  : %d' % len(testing_tweets)            

            # Do Preprocessing
            tweetPreprocessing(testing_tweets, 'Testing')

            # Do Feature Extraction
            featureExtraction(testing_tweets, 'Testing', TRAINING_DICT_FILE)

            # Converting testing data to pandas data frame
            convertFeaturesToDataFrame(testing_tweets).to_excel('Export/features/testing.xlsx', index=False) 
            print 'Exported to Export/features/testing.xlsx\n'

        elif command is '7':
            training_features = pd.read_excel('Export/features/training.xlsx')
            testing_features = pd.read_excel('Export/features/testing.xlsx')

            sub_command = ''
            while sub_command is not 'x':
                print '-----------------------------------------------------------------------'
                print 'Choose Kernel: '
                print '    ----------   Calculate Testing Kernel from Features   ----------'
                print '    1 - Linear'
                print '    2 - Polynomial'
                print '    3 - RBF'
                print '    4 - Sigmoid'
                print '    x - Back to main menu'

                sub_command = raw_input("Enter command : ") 
                for selected_kernel in sub_command.split(","):
                    if selected_kernel is '1':
                        print '\n----------   Calculating Kernel Linear   ----------'

                        linear = calculateTestingKernelLinear(training_features, testing_features)

                        linear.to_excel('Export/kernel/testing/linear.xlsx', sheet_name=str('linear'), index=False)
                        print 'Exported to Export/kernel/testing/linear.xlsx'

                        os.system('say "testing kernel linear generated"')

                    elif selected_kernel is '2':
                        print '\n----------   Calculating Kernel Polynomial   ----------'

                        polynomial = calculateTestingKernelPolynomial(training_features, testing_features)
                        polynomial.to_excel('Export/kernel/testing/polynomial.xlsx', sheet_name=str('polynomial'), index=False)
                        print 'Exported to Export/kernel/testing/polynomial.xlsx'

                        os.system('say "testing kernel polynomial generated"')

                    elif selected_kernel is '3':
                        print '\n----------   Calculating Kernel RBF   ----------'

                        print "Gamma's ", str(gammas)
                        print ''
                        writer = pd.ExcelWriter('Export/kernel/testing/rbf.xlsx')
                        for index_gamma, gamma in enumerate(gammas):
                            rbf = calculateTestingKernelRbf(training_features, testing_features, index_gamma=index_gamma, gamma=float(gamma))
                            rbf.to_excel(writer, sheet_name=str(gamma), index=False)
                        writer.save()
                        print 'Exported to Export/kernel/testing/rbf.xlsx'

                        os.system('say "testing kernel rbf generated"')

                    elif selected_kernel is '4':
                        print '\n----------   Calculating Testing Kernel Sigmoid   ----------'

                        print "a's ", str(aa)
                        print "r's ", str(rr)
                        print ''
                        writer = pd.ExcelWriter('Export/kernel/testing/sigmoid.xlsx')
                        for index_r, r in enumerate(rr):
                            for index_a, a in enumerate(aa):
                                sigmoid = calculateTestingKernelSigmoid(training_features, testing_features, index_a=index_a, index_r=index_r, a=float(a), r=float(r))
                                sigmoid.to_excel(writer, sheet_name=str(a)+';'+str(r), index=False)
                        writer.save()
                        print 'Exported to Export/kernel/testing/sigmoid.xlsx'

                        os.system('say "testing kernel sigmoid generated"')  

                    else:
                        print 'Back to main menu'
                        break

            print ''

        elif command is '8':
            sub_command = ''
            while sub_command is not 'x':
                print '-----------------------------------------------------------------------'
                print 'Choose Kernel: '
                print '    ----------   Classication Testing Data   ----------'
                print '    1 - Linear'
                print '    2 - Polynomial'
                print '    3 - RBF'
                print '    4 - Sigmoid'
                print '    x - Back to main menu'

                sub_command = raw_input("Enter command : ")

                timeObj = time.localtime(time.time())
                timestamp = '%d-%d-%d %d:%d:%d' % (timeObj.tm_mday, timeObj.tm_mon, timeObj.tm_year, timeObj.tm_hour, timeObj.tm_min, timeObj.tm_sec)

                for selected_kernel in sub_command.split(","):
                    if selected_kernel is '1':
                        print '\n----------   Load Training Model Kernel Linear   ----------'
                        print 'Read model training from Export/model/linear.xlsx'
                        model = pd.read_excel('Export/model/linear.xlsx').values.tolist()
                        kernel_type = 'linear'

                    elif selected_kernel is '2':
                        print '\n----------   Load Training Model Kernel Polynomial   ----------'
                        print 'Read model training from Export/model/polynomial.xlsx'
                        model = pd.read_excel('Export/model/polynomial.xlsx').values.tolist()
                        kernel_type = 'polynomial'

                    elif selected_kernel is '3':
                        print '\n----------   Load Training Model Kernel RBF   ----------'
                        print 'Read model training from Export/model/rbf.xlsx'
                        model = pd.read_excel('Export/model/rbf.xlsx').values.tolist()
                        kernel_type = 'rbf'

                    elif selected_kernel is '4':
                        print '\n----------   Load Training Model Kernel Sigmoid   ----------'
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

                        if kernel_type == 'rbf':
                            testing_kernel = master_testing_kernel[model_set[0].getGamma()].values
                        elif kernel_type == 'sigmoid':
                            testing_kernel = master_testing_kernel[model_set[0].getA()+';'+model_set[0].getR()].values
                        else:
                            testing_kernel = master_testing_kernel

                        result_training_model.append(svmClassification(training_model=model_set, testing_kernel=testing_kernel, kernel_type=kernel_type))
                    
                    convertResultToDataFrame(result_training_model).to_excel('Export/result/'+timestamp+' '+kernel_type+'.xlsx', index=False) 
                    print 'Exported to Export/result/'+timestamp+' '+kernel_type+'.xlsx'

                    os.system('say "classification '+kernel_type+' has finished"')  
                
        elif command is 'p':
            print '\n----------   List of Hyperparameter   ----------'
            print 'C          : ', len(Cs), ' ', str(Cs)
            print 'tols       : ', len(tols), ' ', str(tols)
            print 'max_passes : ', str(max_passes)
            print ''
            print 'gamma      : ', len(gammas), ' ', str(gammas)
            print ''
            print 'a          : ', len(aa), ' ', str(aa)
            print 'r          : ', len(rr), ' ', str(rr)
            print ''

    print 'Program terminated'

    

    
