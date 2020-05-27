import util.Machine

from Matrix import Matrix

class LeastSquaresMultiOutput(util.Machine.Machine):
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
        #self.ln = Matrix.addMatrix(self.ln, cost)

        self.runs += 1
        return {
            "predicted": predicted,
            "cost": cost
        }

    def predict(self, inputData):
        inputs = Matrix.vec(matrix = inputData)
        
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
        # Desired and predicted are vectors, so: ||y - yp||^2 = (y - yp)'(y - yp) = the sum of (yi - ypi) ^ 2 for all yi, ypi in y and yp
        cost = Matrix.subMatrix(desired, predicted)
        cost = Matrix.multMatrix(cost, cost)
        cost = sum(cost[0])
        return cost

    def dldw(self, desired, predicted, s):
        # dldw = -2 * (desired - predicted)' * DIAG[Sigmoidal(V' * h(W, s)) *(element by element multiplication)* (1 - Sigmoidal(V' * h(W, s)))] * V * DIAG[Sigmoidal(W * s)] * uk * s'
        difference = Matrix.subMatrix(desired, predicted)
        difference = Matrix.multScalar(matrix = difference, scalar = -2)
        diffTrans = Matrix.transpose(difference)

        vTrans = Matrix.transpose(self.V)

        sTrans = Matrix.transpose(s)

        h = self.hBasis(W = self.W, s = s)
        Vh = Matrix.linearTransform(vTrans, h)
        sigmVh = Matrix.sigmoidal(Vh)
        invSigVh = Matrix.invertNormData(sigmVh)

        sigVhInvSigVh = Matrix.multMatrix(sigmVh, invSigVh)
        diagVh = Matrix.DIAG(sigVhInvSigVh)

        Ws = Matrix.linearTransform(self.W, s)
        sigWs = Matrix.sigmoidal(Ws)
        diagWs = Matrix.DIAG(sigWs)

        diffDiagVh = Matrix.linearTransform(diffTrans, diagVh)
        diffDiagVt = Matrix.linearTransform(diffDiagVh, vTrans) 
        diffDiagVtDiag = Matrix.linearTransform(diffDiagVt, diagWs)

        # Build each row of W
        result = []
        for k in range(self.nHidden):
            
            uk = Matrix.columnOfI(k, self.nHidden)
            ukSt = Matrix.linearTransform(uk, sTrans)

            dcdwk = Matrix.linearTransform(diffDiagVtDiag, ukSt)

            # Extract first and only row from the dcdwk matrix
            dcdwk = Matrix.transpose(dcdwk)
            result.append(dcdwk[0])

        result = Matrix.transpose(result)
        return result

    def dldv(self, desired, predicted, s):
        # dldv = -2 * (desired - predicted)' * DIAG[S(V' * h(W, s)) *(element by element multiplication) * (1 - S(V' * h(W, s)))] * uk * h'
        difference = Matrix.subMatrix(desired, predicted)
        difference = Matrix.multScalar(matrix = difference, scalar = -2)
        diffTrans = Matrix.transpose(difference)

        vTrans = Matrix.transpose(self.V)

        h = self.hBasis(W = self.W, s = s)
        hTrans = Matrix.transpose(h)

        Vh = Matrix.linearTransform(vTrans, h)
        sigmVh = Matrix.sigmoidal(Vh)
        invSigVh = Matrix.invertNormData(sigmVh)

        sigVhInvSigVh = Matrix.multMatrix(sigmVh, invSigVh)
        diagVh = Matrix.DIAG(sigVhInvSigVh)

        diffDiagVh = Matrix.linearTransform(diffTrans, diagVh)
        
        # Build each row of V
        result = []
        for k in range(self.nOutput):
            uk = Matrix.columnOfI(k, self.nOutput)
            ukH = Matrix.linearTransform(uk, hTrans)

            dcdwk = Matrix.linearTransform(diffDiagVh, ukH)

            # Extract first and only row from the dcdwk matrix
            dcdwk = Matrix.transpose(dcdwk)
            result.append(dcdwk[0])

        result = Matrix.transpose(result)

        return result

    def updateW(self, desired, predicted, inputData):
        s = inputData

        derivative = self.dldw(desired = desired, predicted = predicted, s = s)
        stepAmount = Matrix.multScalar(matrix = derivative, scalar = self.learnRate)

        wNew = Matrix.subMatrix(self.W, stepAmount)
        return wNew

    def updateV(self, desired, predicted, inputData):
        s = inputData
        
        derivative = self.dldv(desired = desired, predicted = predicted, s = s)
        stepAmount = Matrix.multScalar(matrix = derivative, scalar = self.learnRate)

        # Transpose stepAmount to fix row alignment issue caused by structure of program
        stepAmount = Matrix.transpose(stepAmount)

        # Calculate the new V
        vNew = Matrix.subMatrix(self.V, stepAmount)
        return vNew