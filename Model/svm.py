class SVM:

    def __init__(self, clas, C, tol, gamma, a, r, label, alpha, bias):
        self.sentiment = self.convertSentiment(clas)
        self.clas = clas
        self.C = C
        self.tol = tol
        self.gamma = gamma
        self.a = a
        self.r = r
        self.clas = clas
        self.label = label
        self.alpha = alpha 
        self.bias = bias

    def getSentiment(self):
        return self.sentiment

    def getClass(self):
        return self.clas

    def getC(self):
        return self.C

    def getTol(self):
        return self.tol

    def getGamma(self):
        return self.gamma

    def getA(self):
        return self.a

    def getR(self):
        return self.r

    def getLabel(self):
        return self.label

    def getAlpha(self):
        return self.alpha

    def getBias(self):
        return self.bias
           
    def convertSentiment(self, clas):
        if clas == 1:
            return 'Positive'
        if clas == 0:
            return 'Neutral'
        if clas == -1:
            return 'Negative'

    def printData(self):
        print '-------------------------------------------------------------------------------------'
        print 'Name     : ', self.sentiment
        print 'Class    : ', self.clas
        print 'C        : ', self.C
        print 'Tol      : ', self.tol
        # print 'Labels   : ', self.label
        # print 'Aplhas   : ', self.alpha
        # print 'Bias     : ', self.bias

    

