import csv
import random

from Matrix import Matrix

class ShadeLearning:

    INPUTS = [
        "x distance from center",
        "y distance from center",
        "N pixel value",
        "NE pixel value",
        "E pixel value",
        "SE pixel value",
        "S pixel value",
        "SW pixel value",
        "W pixel value",
        "NW pixel value"
    ]

    OUTPUTS = [
        "Predicted shade of the pixel"
    ]

    def __init__(self):
        pass

    def learn(self):
        sampleData = self.getSampleData()
        nSamples = len(sampleData["centers"])

        self.sampleWidth = 5

        # to do: add noise to white background of points when they are written into a field

        self.nOutput = 1
        self.nHidden = 6
        self.nInputs = 10
        self.learnRate = 4


        V = Matrix.getRandomWeights(width = self.nOutput, height = self.nHidden)
        W = Matrix.getRandomWeights(width = self.nInputs, height = self.nHidden)

        descend = True
        passCount = 1
        while descend:
            ln = Matrix.getEmptyMatrix(self.sampleWidth, self.sampleWidth)

            for centers in sampleData["centers"]:

                predictions = Matrix.getEmptyMatrix(self.sampleWidth, self.sampleWidth)

                for center in centers:
                    for x, y in self.pixelOrder():
                        xDist = abs(center[0] - x)
                        yDist = abs(center[1] - y)

                        predicted = self.predict(V, W, x, y)
                        
                        V = self.updateV(predicted = predicted, V = V, W = W, x = x, y = y)
                        W = self.updateW(desired = sample[x][y], predicted = predicted, V = V, W = W, x = x, y = y)

                        predictions[x][y] = predicted

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

    def getSampleData(self):
        centerPath = "C:/Users/Cody/Documents/Professional/inkpoint-ml/data/centers.csv"
        shadePath = "C:/Users/Cody/Documents/Professional/inkpoint-ml/data/shades.csv"

        data = {
            "centers": [],
            "shades": {},
            "samples": []
        }
        with open(centerPath) as centerFile:
            centerReader = csv.reader(centerFile)

            for centers in centerReader:
                centerList = []
                for center in centers:
                    parse = center.split(":")
                    pos = [str(parse[0]), str(parse[1])]
                    centerList.append(pos)

                # Store the converted values
                data["centers"].append(centerList)

        with open(shadePath) as shadeFile:
            shadeReader = csv.reader(shadeFile)

            shadeReader.__next__()
            shadeData = shadeReader.__next__()
            
            data["shades"]["avg"] = float(shadeData[0])
            data["shades"]["min"] = float(shadeData[1])
            data["shades"]["max"] = float(shadeData[2])

        return data
        
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