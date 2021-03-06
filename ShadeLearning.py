import csv
import random
import math
import util.machine.LeastSquares as ML

from Matrix import Matrix
from util.ImageSpace import ImageSpace
from PIL import Image

class ShadeLearning:

    # These do nothing - they simply exist as a helpful reference
    INPUTS = [
        "x distance from center",
        "y distance from center",
        "x position",
        "y posposition",
        "x of point's center",
        "y of point's center",
        "shade seed value",
        "numerical seed value",
        [
            "Pixel Values of image that does/will contain ink point",
            "pixel0",
            "pixel1",
            "...",
            "pixel8"
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

        self.machine = None
        self.sampleWidth = 5
        self.nOutput = 1
        self.nHidden = 9
        self.nInputs = 16
        self.learnRate = 0.001

    # The learning algorithm
    def learn(self, save = False):

        sampleData = self.getSampleData()
        nSamples = len(sampleData["samples"])

        # to do: add noise to white background of points when they are written into a field

        if self.machine is None:
            self.machine = ML.LeastSquares(
                nOutput = self.nOutput,
                nHidden = self.nHidden,
                nInputs = self.nInputs,
                learnRate = self.learnRate
            )

        descend = True
        passCount = 1
        while descend:
            ln = Matrix.getEmptyMatrix(self.sampleWidth, self.sampleWidth)
            shadePredictions = None

            indexSeed = 0
            for sampleId in sampleData["samples"]:
                centers = sampleData["samples"][sampleId]["centers"]
                sampleImg = sampleData["samples"][sampleId]["img"]
                sampleNorm = Matrix.pixNorm(matrix = sampleImg)

                predictions = []
                
                for center in centers:
                    xCenter, yCenter = center
                    xCenter = int(xCenter)
                    yCenter = int(yCenter)
                    shadePredictions = Matrix.getEmptyMatrix(self.sampleWidth, self.sampleWidth)
                    cost = Matrix.getEmptyMatrix(self.sampleWidth, self.sampleWidth)

                    shadePredictions[xCenter][yCenter] = sampleNorm[xCenter][yCenter]
                    for x, y in self.pixelOrder([xCenter, yCenter]):
                        if x == xCenter and y == yCenter:
                            continue

                        xDist = abs(int(center[0]) - x)
                        yDist = abs(int(center[1]) - y)

                        inputs = {
                            "xDist": xDist / self.sampleWidth,
                            "yDist": yDist / self.sampleWidth,
                            "xPos": x,
                            "yPos": y,
                            "xCen": xCenter,
                            "yCen": yCenter,
                            "shadeSeed": sampleNorm[xCenter][yCenter], 
                            "numericalSeed": 1 / (indexSeed + 1)
                        }
                        inputs.update(
                            ImageSpace.ExtractNeighbors(
                                x = x,
                                y = y, 
                                imgField = shadePredictions
                            )
                        )
                        
                        resultData = self.machine.learnOnce(desired = sampleNorm[x][y], inputData = inputs)
                        shadePredictions[x][y] = resultData["predicted"]
                        cost[x][y] = resultData["cost"]

                    ln = Matrix.addMatrix(ln, cost)
                    indexSeed += 1

            ln = Matrix.multScalar(matrix = ln, scalar = 1 / nSamples)
            avgObjective = 0
            for x in range(len(ln)):
                for y in range(len(ln[0])):
                    avgObjective += ln[x][y]
            avgObjective = avgObjective / (self.sampleWidth * self.sampleWidth)

            print(" ------------ Pass: " + str(passCount) +  " ------------ ")
            print("Avg  : " + str(avgObjective))
            print("Total: " + str(self.machine.getRisk()))
            print("ln   : ")
            Matrix.printM(ln)
            
            finished = True
            for x in range(self.sampleWidth):
                for y in range(self.sampleWidth):
                    if ln[x][y] > 0.21:
                        finished = False


            if finished or passCount > 10 :
                break

            #print("Continue? (y/n)")
            #cont = input()

            #if cont == "n":
            #    break

            passCount += 1

        self.theta = self.machine.getTheta()
        if save:
            vPath = "C:/Users/Cody/Documents/Professional/inkpoint-ml/data/vWeights.csv"
            wPath = "C:/Users/Cody/Documents/Professional/inkpoint-ml/data/wWeights.csv"
            Matrix.SaveCSV(matrix = self.theta["V"], path = vPath)
            Matrix.SaveCSV(matrix = self.theta["W"], path = wPath)

        print("\nFinished!")

    def getSampleData(self):
        centerPath = "C:/Users/Cody/Documents/Professional/inkpoint-ml/data/centers.csv"
        shadePath = "C:/Users/Cody/Documents/Professional/inkpoint-ml/data/shades.csv"

        data = {
            "centers": {},
            "shades": {},
            "samples": {}
        }

        data = {
            "samples": {},
            "shades": {}
        }
        with open(centerPath) as centerFile:
            centerReader = csv.reader(centerFile)

            for centers in centerReader:
                centerList = []
                index = centers.pop(0)
                data["samples"][str(index)] = {}

                for center in centers:
                    parse = center.split(":")
                    pos = [str(parse[0]), str(parse[1])]
                    centerList.append(pos)

                # Store the converted values
                data["samples"][str(index)]["centers"] = centerList

                # Read the sample image data based on the extracted index
                img = self.readSampleImage(index = index)
                data["samples"][str(index)]["img"] = img

        with open(shadePath) as shadeFile:
            shadeReader = csv.reader(shadeFile)

            shadeReader.__next__()
            shadeData = shadeReader.__next__()
            
            data["shades"]["avg"] = float(shadeData[0])
            data["shades"]["min"] = float(shadeData[1])
            data["shades"]["max"] = float(shadeData[2])

        randomOrder = {}
        keys = list(data["samples"].keys())
        random.shuffle(keys)

        for key in keys:
            randomOrder[key] = data["samples"][key]

        data["samples"] = randomOrder

        return data
        
    def readSampleImage(self, index):
        imgPath = "C:/Users/Cody/Documents/Professional/inkpoint-ml/samples/input" + str(index) + ".png"
        img = Image.open(imgPath)
        imgSpace = ImageSpace(img = img)

        return imgSpace.getField()

    def gen(self, index, baseShade):
        result = Matrix.getEmptyMatrix(self.sampleWidth, self.sampleWidth)

        center = self.chooseCenter()
        xCenter, yCenter = center
        xCenter = int(xCenter)
        yCenter = int(yCenter)

        seedShade = baseShade / 255
        shades = Matrix.getEmptyMatrix(self.sampleWidth, self.sampleWidth)
        #shades[xCenter][yCenter] = seedShade
        for x, y in self.pixelOrder([xCenter, yCenter]):
            #if x == xCenter and y == yCenter:
                #continue

            xDist = abs(int(center[0]) - x)
            yDist = abs(int(center[1]) - y)
            
            inputs = {
                "xDist": xDist / self.sampleWidth,
                "yDist": yDist / self.sampleWidth,
                "xPos": x,
                "yPos": y,
                "xCen": xCenter,
                "yCen": yCenter,
                "shadeSeed": seedShade,
                "numericalSeed": 1 / (index + 1)
            }
            inputs.update(
                ImageSpace.ExtractNeighbors(
                    x = x, 
                    y = y, 
                    imgField = shades
                )
            )

            prediction = self.machine.predict(inputData = inputs)
            shades[x][y] = prediction[0][0]

        shades = Matrix.multScalar(matrix = shades, scalar = 255)
        shades = Matrix.toInteger(matrix = shades)

        return shades
    
    # Generates the order in which pixels should be iterated when processing a sample.
    # The center is always chosen first, and all chosen pixel after that "radiate" out
    # from this center.
    # There is an element of non-determinism so as to prevent memorization, which is
    # implemented via a random shuffling of each layer once it is compiled. Layer ordering
    # is still enforced.
    def pixelOrder(self, center):

        centerX = center[0]
        centerY = center[1]
        order = [ 
            [centerX, centerY] 
        ]
        layers = []

        i = 0
        xMin = centerX - 1
        xMax = centerX + 1
        yMin = centerY - 1
        yMax = centerY + 1
        while True:
            layer = []

            if xMin < 0 and xMax > self.sampleWidth - 1 and yMin < 0 and yMax > self.sampleWidth - 1:
                break

            for x in range(xMin, xMax + 1):
                if x < 0 or x > self.sampleWidth - 1:
                    continue

                if yMin > -1:
                    layer.append([x, yMin])

                if yMax < self.sampleWidth:
                    layer.append([x, yMax])

            for y in range(yMin, yMax + 1):
                if y < 0 or y > self.sampleWidth - 1:
                    continue

                if y == yMin or y == yMax:
                    continue

                if xMin > -1:
                    layer.append([xMin, y])

                if xMax < self.sampleWidth:
                    layer.append([xMax, y])

            random.shuffle(layer)
            layers.append(layer)

            i += 1
            xMin -= 1
            xMax += 1
            yMin -= 1
            yMax += 1

        for i in range(len(layers)):
            order.extend(layers[i])

        return order

    def chooseCenter(self):
        absoluteCenter = (self.sampleWidth - 1) / 2
        aC = absoluteCenter
        prob = { aC - 1: 0.15, aC: 0.75, aC + 1: 1.0 }

        xSet = None
        ran = random.random()
        for x in [aC - 1, aC, aC + 1]:
            if prob[x] > ran:
                xSet = x
                break

        ySet = None
        ran = random.random()
        for y in [aC - 1, aC, aC + 1]:
            if prob[y] > ran:
                ySet = y
                break

        return [x, y]

    def centerShade(self):
        return random.triangular(low = 211, high = 255, mode = 249)

    def load(self):
        path = "C:/Users/Cody/Documents/Professional/inkpoint-ml/data/static/"

        V = Matrix.LoadCSV(path = path + "vWeights.csv")
        W = Matrix.LoadCSV(path = path + "wWeights.csv")

        V = Matrix.stringToFloat(V)
        W = Matrix.stringToFloat(W)

        self.machine = ML.LeastSquares(
            nOutput = self.nOutput,
            nHidden = self.nHidden,
            nInputs = self.nInputs,
            learnRate = self.learnRate
        )

        self.machine.setTheta(V = V, W = W)