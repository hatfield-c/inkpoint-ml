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

    # The learning algorithm
    def learn(self, save = False):

        sampleData = self.getSampleData()
        nSamples = len(sampleData["samples"])
        self.sampleWidth = 5

        # to do: add noise to white background of points when they are written into a field

        if self.machine is None:
            self.machine = ML.LeastSquares(
                nOutput = 1,
                nHidden = 9,
                nInputs = 16,
                learnRate = 0.001
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
                    for x, y in self.pixelOrder():
                        if x == xCenter and y == yCenter:
                            continue

                        xDist = abs(int(center[0]) - x)
                        yDist = abs(int(center[1]) - y)

                        inputs = {
                            "xDist": xDist / 5,
                            "yDist": yDist / 5,
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
                    if ln[x][y] > 0.15:
                        finished = False


            if finished or passCount > 10:
                break

            #print("Continue? (y/n)")
            #cont = input()

            #if cont == "n":
            #    break

            passCount += 1

        self.theta = self.machine.getTheta()
        if save:
            vPath = "C:/Users/hatfi/Documents/professional/inkpoint-ml/data/vWeights.csv"
            wPath = "C:/Users/hatfi/Documents/professional/inkpoint-ml/data/wWeights.csv"
            Matrix.SaveCSV(matrix = self.theta["V"], path = vPath)
            Matrix.SaveCSV(matrix = self.theta["W"], path = wPath)

        print("\nFinished!")

    def getSampleData(self):
        centerPath = "C:/Users/hatfi/Documents/professional/inkpoint-ml/data/centers.csv"
        shadePath = "C:/Users/hatfi/Documents/professional/inkpoint-ml/data/shades.csv"

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
        imgPath = "C:/Users/hatfi/Documents/professional/inkpoint-ml/samples/input" + str(index) + ".png"
        img = Image.open(imgPath)
        imgSpace = ImageSpace(img = img)

        return imgSpace.getField()

    def gen(self, index):
        result = Matrix.getEmptyMatrix(self.sampleWidth, self.sampleWidth)

        center = self.chooseCenter()
        xCenter, yCenter = center
        xCenter = int(xCenter)
        yCenter = int(yCenter)

        seedShade = self.centerShade() / 255
        shades = Matrix.getEmptyMatrix(self.sampleWidth, self.sampleWidth)
        shades[xCenter][yCenter] = seedShade
        for x, y in self.pixelOrder():
            if x == xCenter and y == yCenter:
                continue

            xDist = abs(int(center[0]) - x)
            yDist = abs(int(center[1]) - y)
            
            inputs = {
                "xDist": xDist / 5,
                "yDist": yDist / 5,
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

    def chooseCenter(self):
        prob = { 1: 0.15, 2: 0.75, 3: 1.0 }

        xSet = None
        ran = random.random()
        for x in [1, 2, 3]:
            if prob[x] > ran:
                xSet = x
                break

        ySet = None
        ran = random.random()
        for y in [1, 2, 3]:
            if prob[y] > ran:
                ySet = y
                break

        return [x, y]

    def centerShade(self):
        return random.triangular(low = 211, high = 255, mode = 249)