class Result:

    def __init__(self, kernel, C, tol, gamma, a, r, accuracy):
        self.kernel = kernel
        self.C = C
        self.tol = tol
        self.gamma = gamma
        self.a = a
        self.r = r
        self.accuracy = accuracy
    
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
    
    def getAccuracy(self):
        return self.accuracy