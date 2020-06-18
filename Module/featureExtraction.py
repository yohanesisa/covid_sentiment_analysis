# from helper import *
from Model.tweet import Tweet
from Module.helper import *
from Library.hmmtagger import MainTagger
from Library.barasa import *
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import numpy as np
import sys
import pandas as pd
from collections import OrderedDict

sastrawiFactory = StemmerFactory()
stemmer = sastrawiFactory.create_stemmer()
tagger = MainTagger('Module/Library/resource/Lexicon.trn', 'Module/Library/resource/Ngram.trn', 0, 3, 3, 0, 0, False, 0.2, 0, 500.0, 1)
SentWordNet = read_barasa()

punctuation_dict = {'exclamation': 0, 'question': 0, 'quotation': 0}
word_dict = {}

def featureExtraction(data, type='Training', training_dict_file=None):
    print '----------   ' + type + ' Feature Extraction Start   ----------'
    preparation(data, type, training_dict_file)
    
    for item in data:
        # Punctuation based feature extraction
        item.setPunctuation(punctuationBased(item.getTokens()))
        item.setTokens(removePunctuation(item.getTokens()))

        # POS tag feature extraction
        posTag = posTagger(item.getTokens())
        item.setPosTag(posTag[0])

        # Sentimen score feature extraction
        item.setSentScore(sentimentScore(posTag[1]))

    # Stemming data before unigram TFIDF
    for index, item in enumerate(data):
        sys.stdout.write('\r')
        sys.stdout.write("Progress %d%%" % (float((index+1))/float(len(data))*100))
        sys.stdout.flush()

        stemmed_word = []
        for word in item.getTokens():
            stemmed_word.append(stemmer.stem(word.encode('utf-8')))

        item.setTokens(stemmed_word)

    initWordDict(data, type, training_dict_file)
    for item in data:
        item.setTfidf(countTFIDF(item.getTokens()))

    if type == 'Training':
        print '\n\nTotal tweet  : %d' % len(data)
        print 'Positive : %d' % countSent(data, 1)
        print 'Neutral  : %d' % countSent(data, 0)
        print 'Negative : %d' % countSent(data, -1) 
        print 'Word     : %d \n' % len(word_dict)

        training_dict = OrderedDict()       # Saving punctuation dict and word dict for testing feature extraction later
        training_dict.update({ 'key': [] })
        training_dict.update({ 'value': [] })
        for punctuation in punctuation_dict:
            training_dict['key'].append('Pun-'+punctuation)
            training_dict['value'].append(punctuation_dict[punctuation])
        for word in word_dict:
            training_dict['key'].append('Idf-'+word)
            training_dict['value'].append(word_dict[word]['idf'])
        
        pd.DataFrame(training_dict, columns=training_dict.keys()).to_excel('Export/features/dict.xlsx', index=False)
        print 'Exported to Export/features/dict.xlsx\n'
    else:
        print '\n\nTotal tweet  : %d' % len(data)


def initWordDict(data, type='Training', training_dict_file=None):
    global word_dict

    if type == 'Training':
        word_dict = {}

        for item in data:
            documentWord = []
            for word in item.getTokens():
                if word in word_dict:                           # If already exist in word dict, increment word total occurence
                    word_dict[word]['totalOccurrence'] += 1
                else:                                           # If not exist in word dict, add new word to word dict and count total occurence as 1 but document occurence keep 0 (will be added later)
                    word_dict[word] = {'totalOccurrence': 1, 'docOccurrence': 0}

                if word not in documentWord:                           # If word haven't appeared yet in this document, increment the word document occurence (Document Frequency)
                    word_dict[word]['docOccurrence'] += 1

                documentWord.append(word)                              # add word to list word that already appeared in this document, so word not count more than once in one document.

        for word in word_dict:
            word_dict[word]['idf'] = np.log((float(len(data))/float(word_dict[word]['docOccurrence'])))
            # print '# IDF ' + word + '\t -> log(' + str(len(data)) + '/' + str(word_dict[word]['docOccurrence']) + ') = ' + str(np.log(float(len(data))/float(word_dict[word]['docOccurrence'])))
    else:
        word_dict = {}

        training_word_dict = pd.read_excel(training_dict_file)

        training_word_dict = training_word_dict[training_word_dict['key'] != 'Pun-exclamation']
        training_word_dict = training_word_dict[training_word_dict['key'] != 'Pun-question']
        training_word_dict = training_word_dict[training_word_dict['key'] != 'Pun-quotation']

        training_word_dict = training_word_dict.values.tolist()

        temp = OrderedDict()
        for word in training_word_dict:
            temp.update({ word[0][4:].encode('utf-8'): { 'totalOccurrence': None, 'docOccurrence': None, 'idf': word[1] }})
        
        word_dict = temp

    # print '%20s' % 'Word', '\t', "IDF"
    # for word in word_dict:
    #     print '%20s' % word, '\t', word_dict[word]['idf']

    
def countTFIDF(tweet):
    TF_dict = {}
    for word in tweet:          # Calculating word occurence in a document
        if word in TF_dict:
            TF_dict[word] += 1
        else:
            TF_dict[word] = 1

    for word in TF_dict:  # Calculating TF -> word document
        # print '# TF ' + word + '\t -> ' + str(TF_dict[word]) + '/' + str(len(tweet)) + ' = ' + str(float(TF_dict[word]) / float(len(tweet)))
        TF_dict[word] = float(TF_dict[word]) / float(len(tweet))

    TFIDF_dict = OrderedDict()
    for word in word_dict:  # Calculating TF-IDF
        TFIDF_dict.update({ word: float(0) })

        if word in TF_dict:
            TFIDF_dict[word] = float(TF_dict[word]) * float(word_dict[word]['idf'])
            # print '# TF-IDF ' + word + '\t = \t' + str(float(TF_dict[word])) + '\tx ' + str(float(word_dict[word]['idf']))  + '\t= ' + str(TFIDF_dict[word])
        else:
            TFIDF_dict[word] = float(0) * float(word_dict[word]['idf'])

    return TFIDF_dict


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

    puntuationBased['exclamation'] = average( float(puntuationBased['exclamation']), float(punctuation_dict['exclamation']))
    puntuationBased['question'] = average( float(puntuationBased['question']), float(punctuation_dict['question']))
    puntuationBased['quotation'] = average( float(puntuationBased['quotation']), float(punctuation_dict['quotation']))

    return puntuationBased


def removePunctuation(tweet):
    tweet = list(filter(lambda i: i not in ['!', '?', '\'', '``'], tweet))
    return tweet


def preparation(data, type='Training', training_dict_file=None):
    print('----------   Preparing Feature Extraction    ----------')

    global punctuation_dict

    if type == 'Training':
        punctuation_dict = {'exclamation': 0, 'question': 0, 'quotation': 0}

        for item in data:
            for word in item.getTokens():
                if(word == '!'):
                    punctuation_dict['exclamation'] += 1
                if(word == '?'):
                    punctuation_dict['question'] += 1
                if(word == '\'' or word == '``'):
                    punctuation_dict['quotation'] += 1
    else:
        punctuation_dict = {'exclamation': 0, 'question': 0, 'quotation': 0}

        training_punc_dict = pd.read_excel(training_dict_file)

        punctuation_dict['exclamation'] = int(training_punc_dict.loc[training_punc_dict['key'] == 'Pun-exclamation'].values[0][1])
        punctuation_dict['question'] = int(training_punc_dict.loc[training_punc_dict['key'] == 'Pun-question'].values[0][1])
        punctuation_dict['quotation'] = int(training_punc_dict.loc[training_punc_dict['key'] == 'Pun-quotation'].values[0][1])

    print('Punctuation Dict')
    print('     Exclamation : %d' % punctuation_dict['exclamation'])
    print('     Question    : %d' % punctuation_dict['question'])
    print('     Quotation   : %d' % punctuation_dict['quotation'])


def average(x, y):
    if y == 0:
        return float(0)

    return float(x) / float(y)
