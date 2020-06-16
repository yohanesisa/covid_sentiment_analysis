class Tweet:

    def __init__(self, id, sentence, sentiment):
        self.id = id
        self.sentence = sentence
        self.sentiment = sentiment
        self.tokens = None
        self.punctuation = None
        self.posTag = None
        self.sentScore = None
        self.tfidf = None

    def getId(self):
        return self.id

    def getSentence(self):
        return self.sentence

    def setSentence(self, data):
        self.sentence = data

    def getSentiment(self):
        return self.sentiment

    def getSentimentLabel(self):
        if self.getSentiment() == 'positive':
            return 1
        if self.getSentiment() == 'negative':
            return -1
        if self.getSentiment() == 'neutral':
            return 0
    
    def setTokens(self, data):
        self.tokens = data

    def getTokens(self):
        return self.tokens

    def setPunctuation(self, data):
        self.punctuation = data

    def getPunctuation(self):
        return self.punctuation

    def setPosTag(self, data):
        self.posTag = data

    def getPosTag(self):
        return self.posTag

    def setSentScore(self, data):
        self.sentScore = data

    def getSentScore(self):
        return self.sentScore

    def setTfidf(self, data):
        self.tfidf = data

    def getTfidf(self):
        return self.tfidf

    def printData(self):
        print '-------------------------------------------------------------------------------------'
        print 'ID        : ', self.id
        print 'Sentiment : ', self.sentiment
        print 'Tweet     : ', self.sentence

        if self.tokens:
            print 'Tokens    : ' + str(self.tokens)

        print 'Features'
        if self.punctuation:
            print '# Punctuation: ' + str(self.punctuation)

        if self.posTag:
            print '# POS Tag    : ' + str(self.posTag)

        if self.sentScore:
            print '# Sent Score : \t' + str(self.sentScore['posSentScore']) + '\t\t' + str(self.sentScore['negSentScore'])

        if self.tfidf:
            print '# TF-IDF     : ' + str(self.tfidf)
            # for word in self.tfidf:
            #     print '\t' + word + '\t' + str(self.tfidf[word])

