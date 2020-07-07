class Result:

    def __init__(self, kernel, C, tol, gamma, a, r, pos_true, pos_pred, net_true, net_pred, neg_true, neg_pred, accuracy_score, precision_score, recall_score, f_score, confusion_matrix):
        self.kernel = kernel
        self.C = C
        self.tol = tol
        self.gamma = gamma
        self.a = a
        self.r = r
        self.pos_true = pos_true
        self.pos_pred = pos_pred
        self.net_true = net_true
        self.net_pred = net_pred
        self.neg_true = neg_true
        self.neg_pred = neg_pred
        self.accuracy_score = accuracy_score
        self.precision_score = precision_score
        self.recall_score = recall_score
        self.f_score = f_score
        self.confusion_matrix = confusion_matrix
    
    def getKernel(self):
        return self.kernel
    
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

    def getPosTrue(self):
        return self.pos_true

    def getPosPred(self):
        return self.pos_pred

    def getNetTrue(self):
        return self.net_true

    def getNetPred(self):
        return self.net_pred

    def getNegTrue(self):
        return self.neg_true

    def getNegPred(self):
        return self.neg_pred
    
    def getAccuracyScore(self):
        return self.accuracy_score

    def getPrecisionScore(self):
        return self.precision_score

    def getRecallScore(self):
        return self.recall_score

    def getFScore(self):
        return self.f_score

    def getConfusionMagrix(self):
        return self.confusion_matrix