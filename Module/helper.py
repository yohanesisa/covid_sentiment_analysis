import pandas as pd
from collections import OrderedDict

def countSent(training, sentiment):
    count = 0
    for item in training:
        if(item.getSentiment() == sentiment):
            count = count + 1
    return count

def convertToDataFrame(training):
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
        data['sentimentLabel'].append(item.getSentimentLabel())

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

def printList(data):
    for row in data:
        print row
