from Matrix import Matrix

class Machine:
    def __init__(
            self,
            nOutput, 
            nHidden,
            nInputs,
            learnRate
        ):

        self.nOutput = nOutput
        self.nHidden = nHidden
        self.nInputs = nInputs
        self.learnRate = learnRate

        self.runs = 0
        self.ln = 0
        self.V = Matrix.getRandomWeights(width = self.nOutput, height = self.nHidden)
        self.W = Matrix.getRandomWeights(width = self.nInputs, height = self.nHidden)

    def updateTheta(self, desired, predicted, inputData):
        self.V = self.updateV(desired, predicted, inputData)
        self.W = self.updateW(desired, predicted, inputData)

    def updateW(self, desired, predicted, inputData):
        s = [ list(inputData.values()) ]

        derivative = self.dldw(desired = desired, predicted = predicted, s = s)
        stepAmount = Matrix.multScalar(matrix = derivative, scalar = self.learnRate)

        wNew = Matrix.subMatrix(self.W, stepAmount)
        return wNew

    def updateV(self, desired, predicted, inputData):
        s = [ list(inputData.values()) ]
        
        derivative = self.dldv(desired = desired, predicted = predicted, s = s)
        stepAmount = Matrix.multScalar(matrix = derivative, scalar = self.learnRate)

        # Transpose stepAmount to fix row alignment issue caused by structure of program
        stepAmount = Matrix.transpose(stepAmount)

        # Calculate the new V
        vNew = Matrix.subMatrix(self.V, stepAmount)
        return vNew

    def getRisk(self):
        if self.runs < 1:
            return self.ln
        
        return self.ln / self.runs

    def getTheta(self):
        return {
            "V": self.V,
            "W": self.W
        }

    def setTheta(self, V, W):
        self.V = V
        self.W = W

    def calcCost(self, desired, predicted):
        pass

    def learnOnce(self, desired, inputData):
        pass

    def predict(self, inputData):
        pass

    def hBasis(self, W, s):
        pass

    def dldw(self, desired, predicted, s):
        pass

    def dldv(self, desired, predicted, s):
        pass