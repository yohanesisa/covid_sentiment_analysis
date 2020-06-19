import pandas as pd
from collections import OrderedDict

def countSent(training, sentiment):
    count = 0
    for item in training:
        if(item.getSentiment() == sentiment):
            count = count + 1
    return count

def convertRawToDataFrame(raw):
    data = OrderedDict()
    data.update({ 'id': [] })
    data.update({ 'sentence': [] })
    data.update({ 'sentiment': [] })

    for item in raw:
        data['id'].append(raw[item].getId())
        data['sentence'].append(raw[item].getSentence())
        data['sentiment'].append(raw[item].getSentiment())

    df = pd.DataFrame(data, columns=data.keys())

    return df

def convertTweetsToDataFrame(tweets):
    data = OrderedDict()
    data.update({ 'id': [] })
    data.update({ 'sentence': [] })
    data.update({ 'sentiment': [] })

    for tweet in tweets:
        data['id'].append(tweet.getId())
        data['sentence'].append(tweet.getSentence())
        data['sentiment'].append(tweet.getSentiment())

    df = pd.DataFrame(data, columns=data.keys())

    return df

def convertFeaturesToDataFrame(training):
    data = OrderedDict()
    data.update({ 'sentimentLabel': [] })

    # Init Data
    initData = training[0]
    for punctuation in initData.getPunctuation():
        data.update({ punctuation+'-Punc': [] })
    for sentScore in initData.getSentScore():
        data.update({ sentScore: [] })
    for tag in initData.getPosTag():
        data.update({ tag+'-PosTag': [] })
    for word in initData.getTfidf():
        data.update({ word+'-TFIDF': [] })
    
    # Fill cell with data training
    for item in training:
        data['sentimentLabel'].append(item.getSentiment())

        for punctuation in item.getPunctuation():
            data[punctuation+'-Punc'].append(item.getPunctuation()[punctuation])

        for sentScore in item.getSentScore():
            data[sentScore].append(item.getSentScore()[sentScore])

        for tag in initData.getPosTag():
            data[tag+'-PosTag'].append(item.getPosTag()[tag])

        for word in initData.getTfidf():
            data[word+'-TFIDF'].append(item.getTfidf()[word])

    df = pd.DataFrame(data, columns=data.keys())
    
    return df

def convertTrainingModelToDataFrame(training_model, sentimentTraining):
    data = OrderedDict()
    data.update({ 'kernel': [] })
    data.update({ 'sentiment': [] })
    data.update({ 'class': [] })
    data.update({ 'C': [] })
    data.update({ 'tol': [] })
    data.update({ 'gamma': [] })
    data.update({ 'a': [] })
    data.update({ 'r': [] })
    data.update({ 'bias': [] })

    # Init data
    for kernel in training_model:
        if training_model[kernel] != []:
            initData = training_model[kernel]
            break

    for alpha in range(len(initData[0].getAlpha())):
        data.update({ 'alpha'+str(alpha): [] })

    data['kernel'].append('-')
    data['sentiment'].append('-')
    data['class'].append('-')
    data['C'].append('-')
    data['tol'].append('-')
    data['gamma'].append('-')
    data['a'].append('-')
    data['r'].append('-')
    data['bias'].append('-')
    
    for index, label in enumerate(sentimentTraining):
        data['alpha'+str(index)].append(label)

    # Fill cell with data training
    for kernel in training_model:
        for sent in training_model[kernel]:
            data['kernel'].append(kernel)
            data['sentiment'].append(sent.getSentiment())
            data['class'].append(sent.getClass())
            data['C'].append(sent.getC())
            data['tol'].append(sent.getTol())
            data['gamma'].append(sent.getGamma())
            data['a'].append(sent.getA())
            data['r'].append(sent.getR())
            data['bias'].append(sent.getBias())

            for index, alpha in enumerate(sent.getAlpha()):
                data['alpha'+str(index)].append(alpha)

    df = pd.DataFrame(data, columns=data.keys())

    return df

def convertResultToDataFrame(result_training_model):
    data = OrderedDict()
    data.update({ 'kernel': [] })
    data.update({ 'C': [] })
    data.update({ 'tol': [] })
    data.update({ 'gamma': [] })
    data.update({ 'a': [] })
    data.update({ 'r': [] })

    data.update({ 'pos_true': [] })
    data.update({ 'pos_pred': [] })
    data.update({ 'net_true': [] })
    data.update({ 'net_pred': [] })
    data.update({ 'neg_true': [] })
    data.update({ 'neg_pred': [] })

    data.update({ 'accuracy': [] })
    data.update({ 'precision': [] })
    data.update({ 'recall': [] })
    data.update({ 'f_score': [] })

    for result in result_training_model:
        data['kernel'].append(result.getKernel())
        data['C'].append(result.getC())
        data['tol'].append(result.getTol())
        data['gamma'].append(result.getGamma())
        data['a'].append(result.getA())
        data['r'].append(result.getR())

        data['pos_true'].append(result.getPosTrue())
        data['pos_pred'].append(result.getPosPred())
        data['net_true'].append(result.getNetTrue())
        data['net_pred'].append(result.getNetPred())
        data['neg_true'].append(result.getNegTrue())
        data['neg_pred'].append(result.getNegPred())

        data['accuracy'].append(result.getAccuracyScore())
        data['precision'].append(result.getPrecisionScore())
        data['recall'].append(result.getRecallScore())
        data['f_score'].append(result.getFScore())

    df = pd.DataFrame(data, columns=data.keys())

    return df
    

def printList(data):
    for row in data:
        print row
