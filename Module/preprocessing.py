import re
import string
import demoji
import nltk
import io
import sys
import pandas as pd
from collections import OrderedDict
from nltk.tokenize import word_tokenize
# from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

emoticons = [':)', ':]', '=)', ':-)', ':(', ':[', '=(', ':-(', ':p', ':P', '=P', ':-p', ':-P', ':D', '=D', ':-D', ':o', ':O', ':-o', ':-O', ';)', ';-)', '8-)',
             'B-)', '^_^', '-_-', '>:o', '>:O', ':v', ':3', '8|', 'B|', '8-|', 'B-|', '>:(', ':/', ':\\', ':-/', ':-\\', ':\'(', 'O:)', ':*', ':-*', '<3', '(y)', '(Y)']
escapedEmoticons = [re.escape(x) for x in emoticons]

punctuations = ['#', '$', '%', '(', ')', '*', '+', ',', '.', '/', ':',
                ';', '<', '=', '>', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', '&amp;', '&']

escapedPunctuations = [re.escape(x) for x in punctuations]

slangwords = OrderedDict()
for row in pd.read_excel('Data/preprocessing/slangwords.xlsx').values.tolist():
    slangwords.update({row[0]: row[1]})

# sastrawiFactory = StopWordRemoverFactory()
# stopwords = sastrawiFactory.get_stop_words()

stopwords = []
for row in pd.read_excel('Data/preprocessing/stopwords.xlsx').values.tolist():
    stopwords.append(row[0].encode('utf-8'))

def tweetPreprocessing(tweets, type='Training'):
    for index, item in enumerate(tweets):
        item.setTokens(preprocessing(item.getSentence()))
        sys.stdout.write('\r')
        sys.stdout.write(type+" preprocessing progress %d%%" % (float((index+1))/float(len(tweets))*100))
        sys.stdout.flush()

    for index, item in enumerate(tweets):
        item.setTokens(preprocessingPhase2(item.getTokens()))
        # sys.stdout.write('\r')
        # sys.stdout.write(type+" preprocessing progress %d%%" % (float((index+1))/float(len(tweets))*100))
        # sys.stdout.flush()
    
    print ''

    generateBow(tweets, type)

def generateBow(tweets, type):
    bagOfWord = OrderedDict()
    for index, item in enumerate(tweets):
        for word in item.getTokens():
            if word != '\'' or word != '\"' or word != '``' or word != '-':
                if word in bagOfWord:
                    bagOfWord[word] += 1
                else:
                    bagOfWord.update({ word : 1 })

    data = OrderedDict()
    data.update({ 'word': [] })
    data.update({ 'occurence': [] })
    for word in bagOfWord:
        data['word'].append(word)
        data['occurence'].append(bagOfWord[word])

    if type == 'Training':
        pd.DataFrame(data, columns=data.keys()).to_excel('Export/features/trainingBow.xlsx', index=False)
        print 'Exported to Export/features/trainingBow.xlsx\n'
    else:
        pd.DataFrame(data, columns=data.keys()).to_excel('Export/features/testingBow.xlsx', index=False)
        print 'Exported to Export/features/testingBow.xlsx\n'
    
def preprocessing(tweet):
    clean = caseFolding(tweet)
    clean = removeHashtagUrlMention(clean)
    clean = removeEmoticon(clean)
    clean = removePunctuation(clean)
    clean = wordNormalization(clean)
    clean = removeDuplication(clean)
    clean = tokenization(clean)
    clean = replaceSlangwords(clean)
    clean = removeStopword(clean)
    return clean

def preprocessingPhase2(tokens):
    tweet = ' '.join(tokens)
    clean = caseFolding(tweet)
    clean = removeHashtagUrlMention(clean)
    clean = removeEmoticon(clean)
    clean = removePunctuation(clean)
    clean = wordNormalization(clean)
    clean = removeDuplication(clean)
    clean = tokenization(clean)
    clean = replaceSlangwords(clean)
    clean = removeStopword(clean)
    return clean

def caseFolding(tweet):
    result = tweet.lower()  # convert any capital letters to lowercase
    return result

def removeHashtagUrlMention(tweet):
    tweet = re.sub(r'(\B#\w+)', ' ', tweet)  # remove hashtag
    tweet = re.sub(r'(\B@\w+)', ' ', tweet)  # remove mention
    tweet = re.sub(r'((http(s?):\/\/)+\S*)', ' ', tweet)  # remove url
    return tweet

def removeEmoticon(tweet):
    tweet = demoji.replace(tweet, ' ')  # remove emoji
    tweet = re.sub('|'.join(escapedEmoticons), ' ', tweet)  # remove emoticon
    return tweet

def removePunctuation(tweet):
    tweet = re.sub('|'.join(escapedPunctuations), ' ', tweet)  # remove punctuation
    tweet = re.sub(r'(\')', ' \' ', tweet)
    tweet = re.sub(r'(\")', ' \" ', tweet)
    tweet = re.sub(r'(``)', ' `` ', tweet)
    return tweet

def wordNormalization(tweet):
    tweet = re.sub(r'(\w+ny\b)', r'\1'+'a', tweet)  # handle -ny -> -nya
    tweet = re.sub(r'(\w+n)(k)\b', r'\1'+'g', tweet)  # handle -nk -> -ng
    tweet = re.sub(r'(\w+)(x)\b', r'\1'+'nya', tweet)  # handle -x  -> -nya
    tweet = re.sub(r'(\w+)(z)\b', r'\1'+'s', tweet)  # handle -z  -> -s
    tweet = re.sub(r'(oe)', 'u', tweet)  # handle -oe -> -u
    tweet = re.sub(r'(dj)', 'j', tweet)  # handle -dj -> -j
    tweet = re.sub(u'\u00B2', '2', tweet)  # handle word2 -> word-word
    tweet = re.sub(r'\b(\w+)2\b', r'\1'+'-'+r'\1', tweet)  # handle word2 -> word-word
    return tweet

def removeDuplication(tweet):
    tweet = re.sub(r'([a-z])\1{2,}', r'\1', tweet)  # remove duplication any letters a-z
    return tweet

def tokenization(tweet):
    tweet = word_tokenize(tweet)
    return tweet

def replaceSlangwords(tweet):
    for index, word in enumerate(tweet):
        if(word in slangwords):
            tweet[index] = slangwords[word]
    
    return tweet

def printSlangwords():
    # for index, word in enumerate(tweet):
    #     if(word in slangwords):
    #         tweet[index] = slangwords.get(word)

    print '%15s' % 'Slang', '   ', "Good"
    for word in slangwords:
        print '%15s' % word, '   ', slangwords[word]

    print 'Total', len(slangwords)

def removeStopword(tweet):
    tweet = list(filter(lambda i: i not in stopwords, tweet))
    return tweet
