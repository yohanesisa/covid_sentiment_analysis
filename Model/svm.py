class SVM:

    def __init__(self, y, a, b):
        self.name = self.convertName(y)
        self.y = y
        self.a = a 
        self.b = b
           
    def convertName(self, label):
        if label == 1:
            return 'Positive'
        if label == 0:
            return 'Neutral'
        if label == -1:
            return 'Negative'

    def printData(self):
        print '-------------------------------------------------------------------------------------'
        print 'Name     : ' + str(self.name)
        print 'Label    : ' + str(self.y)
        print 'Aplhas   : ' + str(self.a)
        print 'Beta     : ' + str(self.b)

    

