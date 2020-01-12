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
        self.updateV(desired, predicted, inputData)
        self.updateW(desired, predicted, inputData)

    def updateW(self, desired, predicted, inputData):
        s = [ list(inputData.values()) ]

        derivative = self.dldw(desired = desired, predicted = predicted, V = self.V, W = self.W, s = s)
        stepAmount = Matrix.multScalar(matrix = derivative, scalar = self.learnRate)

        wNew = Matrix.subMatrix(W, stepAmount)
        return wNew

    def updateV(self, desired, predicted, inputData):
        s = [ list(inputData.values()) ]

        derivative = self.dldv(desired = desired, predicted = predicted, V = V, W = W, s = s)
        stepAmount = Matrix.multScalar(matrix = derivative, scalar = self.learnRate)
        
        # Transpose stepAmount to fix row alignment issue caused by structure of program
        stepAmount = Matrix.transpose(stepAmount)

        # Calculate the new V
        vNew = Matrix.subMatrix(V, stepAmount)
        return vNew

    def calcCost(self, sample, predicted):
        pass

    def learnOnce(self, inputData):
        pass

    def predict(self, inputData):
        pass

    def hBasis(self):
        pass

    def dldw(self):
        pass

    def dldv(self):
        pass