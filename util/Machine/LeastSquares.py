import util.Machine

from Matrix import Matrix

class LeastSquares(util.Machine.Machine):
    def __init__(
        self,
        nOutput, 
        nHidden,
        nInputs,
        learnRate
    ):
        super().__init__(
            nOutput = nOutput,
            nHidden = nHidden,
            nInputs = nInputs,
            learnRate = learnRate
        )

    def learnOnce(self, desired, inputData):
        predicted = self.predict(inputData)

        self.updateTheta(desired, predicted, inputData)
        cost = self.calcCost(desired, predicted)
        self.ln += cost

        self.runs += 1
        return {
            "predicted": predicted[0][0],
            "cost": cost
        }

    def predict(self, inputData):
        inputs = [ list(inputData.values()) ]
        
        vTrans = Matrix.transpose(self.V)

        h = self.hBasis(self.W, inputs)
        vH = Matrix.linearTransform(vTrans, h)

        vhSigmoid = Matrix.sigmoidal(vH)
        
        return vhSigmoid

    def hBasis(self, W, s):
        phi = Matrix.linearTransform(W, s)
        h = Matrix.softplus(phi)

        return h

    def calcCost(self, desired, predicted):
        return (desired - predicted[0][0]) ** 2

    def dldw(self, desired, predicted, s):
        # dldw = (- (desired - predicted)) * V' * DIAG[Sigmoidal(W * s)] * uk * s'
        difference = desired - predicted[0][0]
        difference = -1 * difference

        vTrans = Matrix.transpose(self.V)
        diffVt = Matrix.multScalar(matrix = vTrans, scalar = difference)

        sTrans = Matrix.transpose(s)
        rowsW = Matrix.transpose(self.W)

        result = []
        for k in range(self.nHidden):
            wk = Matrix.getRow(matrix = self.W, k = k)
            
            wks = Matrix.linearTransform(wk, s)
            
            sigmaWks = Matrix.sigmoidal(wks)
            diagSigma = Matrix.diag(sigmaWks)
            
            uk = Matrix.columnOfI(k, self.nHidden)
            ukSt = Matrix.linearTransform(uk, sTrans)

            diffVtdiag = Matrix.multScalar(diffVt, diagSigma[0][0])
            dcdwk = Matrix.linearTransform(diffVtdiag, ukSt)

            # Extract first and only row from the dcdwk matrix
            dcdwk = Matrix.transpose(dcdwk)
            result.append(dcdwk[0])

        result = Matrix.transpose(result)
        return result

    def dldv(self, desired, predicted, s):
        # dldv = -(y - yd) * predicted * (1 - predicted) * h'

        difference = (-1) * (desired - predicted[0][0])
        inversePredicted = Matrix.subMatrix([ [1] ] , predicted)
        diffInversePred = Matrix.multScalar(matrix = inversePredicted, scalar = difference)
        
        # Get the term (y - yd)(predicted * (1 - predicted))
        predInversePred = Matrix.linearTransform(predicted, diffInversePred)
        predInversePred = predInversePred[0][0]

        h = self.hBasis(self.W, s)

        hTrans = Matrix.transpose(h)
        result = Matrix.multScalar(matrix = hTrans, scalar = predInversePred)

        return result
