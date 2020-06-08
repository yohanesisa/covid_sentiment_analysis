# from helper import *
from Model.tweet import Tweet
from Module.helper import *
from Library.hmmtagger import MainTagger
from Library.barasa import *
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import numpy as np
import sys

sastrawiFactory = StemmerFactory()
stemmer = sastrawiFactory.create_stemmer()
tagger = MainTagger('Module/Library/resource/Lexicon.trn', 'Module/Library/resource/Ngram.trn', 0, 3, 3, 0, 0, False, 0.2, 0, 500.0, 1)
SentWordNet = read_barasa()

punctuation_dict = {'exclamation': 0, 'question': 0, 'quotation': 0}
word_dict = {}

def featureExtraction(training):
    preparation(training)
    print('----------   Feature Extraction Start   ----------')

    for item in training:
        # Punctuation based feature extraction
        item.setPunctuation(punctuationBased(item.getTokens()))
        item.setTokens(removePunctuation(item.getTokens()))

        # POS tag feature extraction
        posTag = posTagger(item.getTokens())
        item.setPosTag(posTag[0])

        # Sentimen score feature extraction
        item.setSentScore(sentimentScore(posTag[1]))

    # Stemming data before unigram TFIDF
    for index, item in enumerate(training):
        sys.stdout.write('\r')
        sys.stdout.write("Progress %d%%" % (float((index+1))/float(len(training))*100))
        sys.stdout.flush()

        stemmed_word = []
        for word in item.getTokens():
            stemmed_word.append(stemmer.stem(word.encode('utf-8')))

        item.setTokens(stemmed_word)

    initWordDict(training)
    for item in training:
        item.setTfidf(countTFIDF(item.getTokens()))

    print('\n\nTotal tweet  : %d' % len(training))
    print('Positive : %d' % countSent(training, 'positive'))
    print('Negative : %d' % countSent(training, 'negative'))
    print('Neutral  : %d' % countSent(training, 'neutral'))
    print('Word     : %d \n' % len(word_dict))


def sentimentScore(tweet):
    for word in tweet:
        if tweet[word] == 'JJ' or tweet[word] == 'CDC' or tweet[word] == 'CDI' or tweet[word] == 'CDO' or tweet[word] == 'CDP' or tweet[word] == 'IN':
            tweet[word] = 'a'  # adverb
        elif tweet[word] == 'VBI' or tweet[word] == 'VBT':
            tweet[word] = 'v'  # verb
        elif tweet[word] == 'RB' or tweet[word] == 'NEG' or tweet[word] == 'SC':
            tweet[word] = 'r'  # additional adverb
        elif tweet[word] == 'NN' or tweet[word] == 'NNP' or tweet[word] == 'NNG' or tweet[word] == 'FW' or tweet[word] == 'MD' or tweet[word] == 'WP':
            tweet[word] = 'n'  # noun
        else:
            tweet[word] = 'n'

    posScore = 0.0
    negScore = 0.0
    for word in tweet:
        posWord = 0.0
        negWord = 0.0
        synsets = SentWordNet[word]

        cleanSynsets = []
        for sent in synsets:
            if sent.synset.endswith(tweet[word]):
                cleanSynsets.append(sent)

        for sent in cleanSynsets:
            posWord += float(sent.pos)
            negWord += float(sent.neg)

        posWord = average(float(posWord),float(len(cleanSynsets)))
        negWord = average(float(negWord),float(len(cleanSynsets)))

        # print word + '-(' + tweet[word] + ') \tpos= ' + str(posWord) + '\tneg= ' + str(negWord)

        posScore += float(posWord)
        negScore += float(negWord)

    # print '---- SUM ---- \tpos= ' + str(posScore) + '\tneg= ' + str(negScore)
        # for sent in cleanSynsets:
        #     print '\t' + str(sent)

    return { 'posSentScore': posScore, 'negSentScore': negScore}

def initWordDict(training):
    for item in training:
        scrap = []
        for word in item.getTokens():
            if word in word_dict:
                word_dict[word]['totalOccurrence'] += 1
            else:
                word_dict[word] = {'totalOccurrence': 1, 'docOccurrence': 0}

            if word not in scrap:
                word_dict[word]['docOccurrence'] += 1

            scrap.append(word)

    for word in word_dict:
        word_dict[word]['idf'] = np.log((float(len(training))/float(word_dict[word]['docOccurrence'])))
        # print '# IDF ' + word + '\t -> log(' + str(len(training)) + '/' + str(word_dict[word]['docOccurrence']) + ') = ' + str(float(len(training))/float(word_dict[word]['docOccurrence']))

    # print '\n Word \t \t Total \t Document \t IDF'
    # for word in word_dict:
    #     print word + '\t \t' + str(word_dict[word]['totalOccurrence']) + '\t' + str(word_dict[word]['docOccurrence']) + '\t \t' + str(word_dict[word]['idf'])


def countTFIDF(tweet):
    TF_dict = {}
    for word in tweet:
        if word in TF_dict:
            TF_dict[word] += 1
        else:
            TF_dict[word] = 1

    for word in TF_dict:  # Calculating TF
        # print '# TF ' + word + '\t -> ' + str(TF_dict[word]) + '/' + str(len(word_dict)) + ' = ' + str(float(TF_dict[word]) / float(len(word_dict)))
        TF_dict[word] = float(TF_dict[word]) / float(len(word_dict))

    TFIDF_dict = {}
    for word in word_dict:  # Calculating TF-IDF
        if word in TF_dict:
            TFIDF_dict[word] = float(TF_dict[word]) * \
                float(word_dict[word]['idf'])
        else:
            TFIDF_dict[word] = float(0)

    return TFIDF_dict


def posTagger(tweet):
    dictionaryTag = {'JJ': 0, 'RB': 0, 'NN': 0, 'NNP': 0, 'NNPP': 0,
                     'NNG': 0, 'VBI': 0, 'VBT': 0, 'IN': 0, 'MD': 0,
                     'CC': 0, 'SC': 0, 'DT': 0, 'UH': 0, 'CDO': 0,
                     'CDC': 0, 'CDP': 0, 'CDI': 0, 'PRP': 0, 'WP': 0,
                     'PRN': 0, 'PRL': 0, 'NEG': 0, 'SYM': 0, 'RP': 0,
                     'FW': 0, }

    taged = tagger.taggingStr(' '.join(tweet))
    cleanTag = {}

    for word in taged:
        tag = word.split('/')
        # print str(tag[0]) + ' -> ' + str(tag[1])
        dictionaryTag[tag[1]] += 1

        cleanTag.update({tag[0]: tag[1]})

    return dictionaryTag, cleanTag


def punctuationBased(tweet):
    puntuationBased = {'exclamation': 0, 'question': 0, 'quotation': 0}

    for word in tweet:
        if(word == '!'):
            puntuationBased['exclamation'] += 1
        if(word == '?'):
            puntuationBased['question'] += 1
        if(word == '\'' or word == '``'):
            puntuationBased['quotation'] += 1

    puntuationBased['exclamation'] = average(
        float(puntuationBased['exclamation']), float(punctuation_dict['exclamation']))
    puntuationBased['question'] = average(
        float(puntuationBased['question']), float(punctuation_dict['question']))
    puntuationBased['quotation'] = average(
        float(puntuationBased['quotation']), float(punctuation_dict['quotation']))
    return puntuationBased


def removePunctuation(tweet):
    tweet = list(filter(lambda i: i not in ['!', '?', '\'', '``'], tweet))
    return tweet


def preparation(training):
    print('----------   Preparing Feature Extraction    ----------')

    for item in training:
        for word in item.getTokens():
            if(word == '!'):
                punctuation_dict['exclamation'] += 1
            if(word == '?'):
                punctuation_dict['question'] += 1
            if(word == '\'' or word == '``'):
                punctuation_dict['quotation'] += 1

    print('Punctuation Dict')
    print('     Exclamation : %d' % punctuation_dict['exclamation'])
    print('     Question    : %d' % punctuation_dict['question'])
    print('     Quotation   : %d' % punctuation_dict['quotation'])


def average(x, y):
    if y == 0:
        return float(0)
    return float(x) / float(y)
