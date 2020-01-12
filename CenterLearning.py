import csv
import math

from Matrix import Matrix

class CenterLearning:

    def __init__(self):
        pass

    def learn(self):
        samples = self.getSampleData()
        nSamples = len(samples)

        self.sampleWidth = 5

        # to do: add noise to white background of points when they are written into a field

        self.nOutput = 1
        self.nHidden = 13
        self.nInputs = 25
        self.learnRate = 4


        V = Matrix.getRandomWeights(width = self.nOutput, height = self.nHidden)
        W = Matrix.getRandomWeights(width = self.nInputs, height = self.nHidden)

        predictions = []
        descend = True
        passCount = 1
        while descend:
            ln = Matrix.getEmptyMatrix(5, 5)

            for sample in samples:

                predictions = []
                for x in range(self.sampleWidth):
                    column = []
                    for y in range(self.sampleWidth):
                        predicted = self.predict(V, W, x, y)
                        column.append(predicted[0][0])
                        
                        V = self.updateV(predicted = predicted, V = V, W = W, x = x, y = y)
                        W = self.updateW(desired = sample[x][y], predicted = predicted, V = V, W = W, x = x, y = y)

                    predictions.append(column)

                cost = self.calcCost(sample, predictions)
                ln = Matrix.addMatrix(ln, cost)

            ln = Matrix.multScalar(matrix = ln, scalar = 1 / nSamples)
            avgObjective = 0
            for x in range(len(ln)):
                for y in range(len(ln[0])):
                    avgObjective += ln[x][y]
            avgObjective = avgObjective / (self.sampleWidth * self.sampleWidth)

            print(" ------------ Pass: " + str(passCount) + "   |   Obj: " + str(avgObjective) + " ------------ ")
            print("ln:")
            Matrix.printM(ln)
            #predictions = Matrix.log(predictions)
            print("probs:")
            Matrix.printM(predictions)
            
            finished = True
            for x in range(self.sampleWidth):
                for y in range(self.sampleWidth):
                    if ln[x][y] > 0.6:
                        finished = False

            if finished:
                break

            passCount += 1

        #Matrix.printM(V)
        #Matrix.printM(W)
        print("\nFinished!")

    def updateV(self, predicted, V, W, x, y):
        derivative = self.dldv(predicted = predicted, V = V, W = W, x = x, y = y)
        stepAmount = Matrix.multScalar(matrix = derivative, scalar = self.learnRate)
        
        # Correction of stepAmount row alignment caused by structure of program
        stepAmount = Matrix.transpose(stepAmount)
        vNew = Matrix.subMatrix(V, stepAmount)
        return vNew

    def updateW(self, desired, predicted, V, W, x, y):
        derivative = self.dldw(desired = desired, predicted = predicted, V = V, W = W, x = x, y = y)
        stepAmount = Matrix.multScalar(matrix = derivative, scalar = self.learnRate)

        wNew = Matrix.subMatrix(W, stepAmount)
        return wNew

    def predict(self, V, W, x, y):
        inputs = Matrix.getEmptyMatrix(self.sampleWidth, self.sampleWidth)
        inputs[x][y] = 1

        vTrans = Matrix.transpose(V)
        h = self.hBasis(W, inputs)
        vH = Matrix.linearTransform(vTrans, h)

        vhSigmoid = Matrix.sigmoidal(vH)
        
        return vhSigmoid

    def hBasis(self, W, sample):
        sampleVec = Matrix.vec(sample)
        phi = Matrix.linearTransform(W, sampleVec)

        h = Matrix.softplus(phi)

        return h

    def dldw(self, desired, predicted, V, W, x, y):
        # dldw = (- (desired - predicted)) * V' * DIAG[Sigmoidal(W * input)] * uk * input'
        difference = desired - predicted[0][0]
        difference = -1 * difference

        vTrans = Matrix.transpose(V)
        diffVt = Matrix.multScalar(matrix = vTrans, scalar = difference)

        inputs = Matrix.getEmptyMatrix(self.sampleWidth, self.sampleWidth)
        inputs[x][y] = 1
        inputs = Matrix.vec(inputs)
        inputsTrans = Matrix.transpose(inputs)

        rowsW = Matrix.transpose(W)

        result = []
        for k in range(self.nHidden):
            wk = Matrix.getRow(matrix = W, k = k)
            
            wks = Matrix.linearTransform(wk, inputs)
            
            sigmaWks = Matrix.sigmoidal(wks)
            diagSigma = Matrix.diag(sigmaWks)
            
            uk = Matrix.columnOfI(k, self.nHidden)
            ukSt = Matrix.linearTransform(uk, inputsTrans)

            diffVtdiag = Matrix.multScalar(diffVt, diagSigma[0][0])
            dcdwk = Matrix.linearTransform(diffVtdiag, ukSt)

            # Extract first and only row from the dcdwk matrix
            dcdwk = Matrix.transpose(dcdwk)
            result.append(dcdwk[0])

        result = Matrix.transpose(result)
        return result
        

    def dldv(self, predicted, V, W, x, y):
        # dldv = predicted * (1 - predicted) * h'

        inversePredicted = Matrix.subMatrix([ [1] ] , predicted)
        
        # Get the term (predicted * (1 - predicted)), and then extract the scalar value
        # from the resulting 1x1 matrix
        predInversePred = Matrix.linearTransform(predicted, inversePredicted)
        predInversePred = predInversePred[0][0]
        
        inputs = Matrix.getEmptyMatrix(self.sampleWidth, self.sampleWidth)
        inputs[x][y] = 1
        h = self.hBasis(W, inputs)

        hTrans = Matrix.transpose(h)
        result = Matrix.multScalar(matrix = hTrans, scalar = predInversePred)

        return result

    def calcCost(self, sample, predicted):
        costs = []
        for x in range(len(sample)):
            column = []
            for y in range(len(sample[0])):
                desired = sample[x][y]
                pred = predicted[x][y]
                #prob = self.calcProb(predicted[x][y])

                cost = (desired * math.log(pred)) + ((1 - desired) * math.log(1 - pred))
                cost = (-1) * cost

                column.append(cost)

            costs.append(column)

        return costs

    def calcProb(self, predicted):
        return 1 / (1 + math.exp((-1) * predicted))

    def getSampleData(self):
        filePath = "C:/Users/Cody/Documents/Professional/inkpoint-ml/data/centers.csv"

        samples = []
        with open(filePath) as sampleFile:
            sampleReader = csv.reader(sampleFile)

            for sample in sampleReader:
                # Convert the center values from the samples into integers
                x = int(sample[0])
                y = int(sample[1])

                # Build a sample result matrix, where pixels that are the center
                # have a value of '1' and pixels which are not the center have a
                # value of '0'
                sampleMatrix = Matrix.getEmptyMatrix(5, 5)
                sampleMatrix[x][y] = 1

                # Store the converted values
                samples.append(sampleMatrix)

        return samples
        
    