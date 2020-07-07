import pandas as pd
from collections import OrderedDict
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

feature_punc = True
feature_sentscore = False
feature_postag = True
feature_unigram_tfidf = True
feature_bow = False

lexicon_labeling = True

def countSent(data, sentiment):
    count = 0
    for item in data:
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

def convertFeaturesToDataFrame(tweets):
    data = OrderedDict()
    data.update({ 'sentimentLabel': [] })

    y_true = []
    y_pred = []

    # Init Data
    initData = tweets[0]
    if feature_punc:
        for punctuation in initData.getPunctuation():
            data.update({ punctuation+'-Punc': [] })
    if feature_sentscore:
        for sentScore in initData.getSentScore():
            data.update({ sentScore: [] })
    if feature_postag:
        for tag in initData.getPosTag():
            data.update({ tag+'-PosTag': [] })
    if feature_unigram_tfidf:
        for word in initData.getTfidf():
            data.update({ word+'-TFIDF': [] })
    if feature_bow:
        for word in initData.getBow():
            data.update({ word+'-BoW': [] })
            
    # Fill cell with data training
    for item in tweets:
        y_true.append(item.getSentiment())
        y_pred.append(item.getPolarity())

        if lexicon_labeling:
            data['sentimentLabel'].append(item.getPolarity())
            # data['sentimentLabel'].append(item.getSentiment())
            # data['polarityLabel'].append(item.getPolarity())
        else:
            data['sentimentLabel'].append(item.getSentiment())

        if feature_punc:
            for punctuation in item.getPunctuation():
                data[punctuation+'-Punc'].append(item.getPunctuation()[punctuation])

        if feature_sentscore:
            for sentScore in item.getSentScore():
                data[sentScore].append(item.getSentScore()[sentScore])

        if feature_postag:
            for tag in initData.getPosTag():
                data[tag+'-PosTag'].append(item.getPosTag()[tag])

        if feature_unigram_tfidf:
            for word in initData.getTfidf():
                data[word+'-TFIDF'].append(item.getTfidf()[word])

        if feature_bow:
            for word in initData.getBow():
                data[word+'-BoW'].append(item.getBow()[word])

    if lexicon_labeling:
        print ''
        print 'Lexicon Tagging Enabled -> Accuracy',accuracy_score(y_true, y_pred)*100

        # accuracy = accuracy_score(y_true, y_pred)
        # precission = precision_score(y_true, y_pred, average='weighted')
        # recall = recall_score(y_true, y_pred, average='weighted')
        # f = f1_score(y_true, y_pred, average='weighted')
        # c_matrix = confusion_matrix(y_true, y_pred, labels=[1, 0, -1])

        # print accuracy, precission, recall, f
        # print c_matrix
                
    df = pd.DataFrame(data, columns=data.keys())
    
    return df

def convertTrainingModelToDataFrame(model, label):
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
    for kernel in model:
        if model[kernel] != []:
            initData = model[kernel]
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
    
    for index, item in enumerate(label):
        data['alpha'+str(index)].append(item)

    # Fill cell with data training
    for kernel in model:
        for sent in model[kernel]:
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

def convertResultToDataFrame(results):
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
    data.update({ 'confusion_matrix': [] })

    for result in results:
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
        data['confusion_matrix'].append(result.getConfusionMagrix())

    df = pd.DataFrame(data, columns=data.keys())

    return df

