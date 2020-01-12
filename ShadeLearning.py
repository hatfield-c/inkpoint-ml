import csv
import random
import math

from Matrix import Matrix
from util.ImageSpace import ImageSpace
from PIL import Image

class ShadeLearning:

    # These do nothing - they simply exist as a helpful reference
    INPUTS = [
        "x distance from center",
        "y distance from center",
        "seed value",
        [
            "Pixel Values of image that does/will contain ink point",
            "pixel0",
            "pixel1",
            "...",
            "pixel24"
        ]
    ]

    OUTPUTS = [
        "Predicted shade of the pixel"
    ]

    def __init__(self):
        self.theta = {
            "V": None,
            "W": None
        }

    # The learning algorithm
    def learn(self, save = False):
        sampleData = self.getSampleData()
        nSamples = len(sampleData["centers"])

        self.sampleWidth = 5

        # to do: add noise to white background of points when they are written into a field

        self.nOutput = 1
        self.nHidden = 7
        self.nInputs = 11
        self.learnRate = 0.001

        V = Matrix.getRandomWeights(width = self.nOutput, height = self.nHidden)
        W = Matrix.getRandomWeights(width = self.nInputs, height = self.nHidden)

        descend = True
        passCount = 1
        while descend:
            ln = Matrix.getEmptyMatrix(self.sampleWidth, self.sampleWidth)
            shadePredictions = None
            indexSeed = 0

            for centerIndex in sampleData["centers"]:
                centers = sampleData["centers"][centerIndex]
                sampleImg = sampleData["samples"][centerIndex]
                sampleNorm = Matrix.pixNorm(matrix = sampleImg)

                predictions = []
                
                for center in centers:
                    xCenter, yCenter = center
                    xCenter = int(xCenter)
                    yCenter = int(yCenter)
                    shadePredictions = Matrix.getEmptyMatrix(self.sampleWidth, self.sampleWidth)

                    for x, y in self.pixelOrder():
                        xDist = abs(int(center[0]) - x)
                        yDist = abs(int(center[1]) - y)

                        inputs = {
                            "x": xDist / 5,
                            "y": yDist / 5,
                            "seed": 1 / (indexSeed + 1)
                        }
                        inputs.update(
                            ImageSpace.ExtractNeighbors(
                                x = x, 
                                y = y, 
                                imgField = shadePredictions
                            )
                        )

                        predictedShade = self.predict(
                            V = V, 
                            W = W, 
                            s = inputs
                        )
                        # Extract the only element from the resulting matrix
                        shadePredictions[x][y] = predictedShade[0][0]
                        
                        V = self.updateV(
                            desired = sampleNorm[x][y],
                            predicted = predictedShade, 
                            V = V, 
                            W = W, 
                            s = inputs
                        )
                        W = self.updateW(
                            desired = sampleNorm[x][y], 
                            predicted = predictedShade, 
                            V = V, 
                            W = W, 
                            s = inputs
                        )

                    cost = self.calcCost(sampleNorm, shadePredictions)
                    ln = Matrix.addMatrix(ln, cost)
                    indexSeed += 1

            ln = Matrix.multScalar(matrix = ln, scalar = 1 / nSamples)
            avgObjective = 0
            for x in range(len(ln)):
                for y in range(len(ln[0])):
                    avgObjective += ln[x][y]
            avgObjective = avgObjective / (self.sampleWidth * self.sampleWidth)

            print(" ------------ Pass: " + str(passCount) + "   |   Obj: " + str(avgObjective) + " ------------ ")
            print("ln:")
            Matrix.printM(ln)
            
            finished = True
            for x in range(self.sampleWidth):
                for y in range(self.sampleWidth):
                    if ln[x][y] > 0.095:
                        finished = False

            if finished:
                break

            passCount += 1

        self.theta["V"] = V
        self.theta["W"] = W

        if save:
            vPath = "C:/Users/Cody/Documents/Professional/inkpoint-ml/data/vWeights.csv"
            wPath = "C:/Users/Cody/Documents/Professional/inkpoint-ml/data/wWeights.csv"
            Matrix.SaveCSV(matrix = self.theta["V"], path = vPath)
            Matrix.SaveCSV(matrix = self.theta["W"], path = wPath)

        print("\nFinished!")

    def predict(self, V, W, s):
        inputs = [ list(s.values()) ]
        
        vTrans = Matrix.transpose(V)
        h = self.hBasis(W, inputs)
        vH = Matrix.linearTransform(vTrans, h)

        vhSigmoid = Matrix.sigmoidal(vH)
        
        return vhSigmoid

    def updateW(self, desired, predicted, V, W, s):
        s = [ list(s.values()) ]

        derivative = self.dldw(desired = desired, predicted = predicted, V = V, W = W, s = s)
        stepAmount = Matrix.multScalar(matrix = derivative, scalar = self.learnRate)

        wNew = Matrix.subMatrix(W, stepAmount)
        return wNew

    def updateV(self, desired, predicted, V, W, s):
        s = [ list(s.values()) ]

        derivative = self.dldv(desired = desired, predicted = predicted, V = V, W = W, s = s)
        stepAmount = Matrix.multScalar(matrix = derivative, scalar = self.learnRate)
        
        # Transpose stepAmount to fix row alignmentissue  caused by structure of program
        stepAmount = Matrix.transpose(stepAmount)

        # Calculate the new V
        vNew = Matrix.subMatrix(V, stepAmount)
        return vNew

    def dldw(self, desired, predicted, V, W, s):
        # dldw = (- (desired - predicted)) * V' * DIAG[Sigmoidal(W * s)] * uk * s'
        difference = desired - predicted[0][0]
        difference = -1 * difference

        vTrans = Matrix.transpose(V)
        diffVt = Matrix.multScalar(matrix = vTrans, scalar = difference)

        sTrans = Matrix.transpose(s)
        rowsW = Matrix.transpose(W)

        result = []
        for k in range(self.nHidden):
            wk = Matrix.getRow(matrix = W, k = k)
            
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

    def dldv(self, desired, predicted, V, W, s):
        # dldv = (y - yd)) * predicted * (1 - predicted) * h'

        difference = (-1) * (desired - predicted[0][0])
        inversePredicted = Matrix.subMatrix([ [1] ] , predicted)
        diffInversePred = Matrix.multScalar(matrix = inversePredicted, scalar = difference)
        
        # Get the term (predicted * (1 - predicted)), and then extract the scalar value
        # from the resulting 1x1 matrix
        predInversePred = Matrix.linearTransform(predicted, diffInversePred)
        predInversePred = predInversePred[0][0]

        h = self.hBasis(W, s)

        hTrans = Matrix.transpose(h)
        result = Matrix.multScalar(matrix = hTrans, scalar = predInversePred)

        return result

    def hBasis(self, W, s):
        phi = Matrix.linearTransform(W, s)
        h = Matrix.softplus(phi)

        return h

    def calcCost(self, sample, predicted):
        costs = []

        #print(predicted)
        for x in range(len(sample)):
            column = []
            for y in range(len(sample[0])):
                desired = sample[x][y]
                pred = predicted[x][y]

                cost = (desired - pred) ** 2
                column.append(cost)

            costs.append(column)

        return costs

    def getSampleData(self):
        centerPath = "C:/Users/Cody/Documents/Professional/inkpoint-ml/data/centers.csv"
        shadePath = "C:/Users/Cody/Documents/Professional/inkpoint-ml/data/shades.csv"

        data = {
            "centers": {},
            "shades": {},
            "samples": {}
        }
        with open(centerPath) as centerFile:
            centerReader = csv.reader(centerFile)

            for centers in centerReader:
                centerList = []
                index = centers.pop(0)
                img = self.readSampleImage(index = index)
                data["samples"][index] = img

                for center in centers:
                    parse = center.split(":")
                    pos = [str(parse[0]), str(parse[1])]
                    centerList.append(pos)

                # Store the converted values
                data["centers"][index] = centerList

                # Read the sample image data based on the extracted index
                img = self.readSampleImage(index = index)
                data["samples"][index] = img

        with open(shadePath) as shadeFile:
            shadeReader = csv.reader(shadeFile)

            shadeReader.__next__()
            shadeData = shadeReader.__next__()
            
            data["shades"]["avg"] = float(shadeData[0])
            data["shades"]["min"] = float(shadeData[1])
            data["shades"]["max"] = float(shadeData[2])

        return data
        
    def readSampleImage(self, index):
        imgPath = "C:/Users/Cody/Documents/Professional/inkpoint-ml/samples/input" + str(index) + ".png"
        img = Image.open(imgPath)
        imgSpace = ImageSpace(img = img)

        return imgSpace.getField()
    
    # Generates the order in which pixels should be iterated when processing a sample.
    # There is an element of non-determinism so as to prevent memorization
    def pixelOrder(self):
        order = [ 
            [2, 2] 
        ]

        firstLayer = [
            [1, 1],
            [1, 2],
            [1, 3],
            [3, 1],
            [3, 2],
            [3, 3],
            [2, 1],
            [2 ,3]
        ]

        secondLayer = [
            [0, 0],
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],
            [4, 0],
            [4, 1],
            [4, 2],
            [4, 3],
            [4, 4],
            [1, 0],
            [2, 0],
            [3, 0],
            [1, 4],
            [2, 4],
            [3, 4]
        ]
        
        random.shuffle(firstLayer)
        random.shuffle(secondLayer)

        order.extend(firstLayer)
        order.extend(secondLayer)

        return order

    def gen(self, index):
        V = self.theta["V"]
        W = self.theta["W"]
        result = Matrix.getEmptyMatrix(self.sampleWidth, self.sampleWidth)

        center = self.chooseCenter()
        xCenter, yCenter = center
        xCenter = int(xCenter)
        yCenter = int(yCenter)

        shades = Matrix.getEmptyMatrix(self.sampleWidth, self.sampleWidth)
        for x, y in self.pixelOrder():
            xDist = abs(int(center[0]) - x)
            yDist = abs(int(center[1]) - y)

            inputs = {
                "x": xDist / 5,
                "y": yDist / 5,
                "index": 1 / (index + 1)
            }
            inputs.update(
                ImageSpace.ExtractNeighbors(
                    x = x, 
                    y = y, 
                    imgField = shades
                )
            )

            prediction = self.predict(V = V, W = W, s = inputs)
            shades[x][y] = prediction[0][0]

        shades = Matrix.multScalar(matrix = shades, scalar = 255)
        shades = Matrix.toInteger(matrix = shades)

        return shades

    def chooseCenter(self):
        choices = [
            [1, 1],
            [1, 2],
            [1, 2],
            [1, 3],
            [2, 1],
            [2, 1],
            [2, 2],
            [2, 2],
            [2, 2],
            [2, 2],
            [2, 3],
            [2, 3],
            [3, 1],
            [3, 2],
            [3, 2],
            [3, 3]
        ]

        return random.choice(choices)