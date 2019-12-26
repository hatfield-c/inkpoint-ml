import csv
import os
import statistics

from PIL import Image
from util.ImageSpace import ImageSpace

class TrainingFactory:
    def __init(self):
        pass

    def buildTraining(self):
        appPath = "C:/Users/Cody/Documents/Professional/inkpoint-ml"
        rootPath = "C:/Users/Cody/Documents/Professional/inkpoint-ml/samples"
        samplePaths = os.listdir(rootPath)

        x = 0
        centerList = []
        shades = {
            "avg": [],
            "max": [],
            "min": []
        }
        for path in samplePaths:
            if path == 'old' or path == 'orig':
                continue
            
            # Extract the file's index from the filename, and store it so it can be used
            # for debugging.
            index = path.split('.')
            index = index[0].split('input')
            index = index[1]

            img = Image.open(rootPath + "/" + path)
            imgSpace = ImageSpace(img = img)

            imgField = imgSpace.getField()

            centers = self.extractCenter(imgField, path)
            self.extractShadeData(shades, centers, imgField)
            centers.insert(0, index)

            centerList.append(centers)

        with open(appPath + '/data/centers.csv', mode = "w", newline = '') as centerFile:
            writer = csv.writer(centerFile)

            for centers in centerList:
                writer.writerow(centers)

        with open(appPath + '/data/shades.csv', mode = "w", newline = '') as centerFile:
            writer = csv.writer(centerFile)

            avgShade = statistics.mean(shades["avg"])
            minShade = min(shades["min"])
            maxShade = max(shades["max"])
            
            labels = ["avg", "min", "max"]
            data = [avgShade, minShade, maxShade]
            
            writer.writerow(labels)
            writer.writerow(data)

    def extractCenter(self, imgField, path):
        width = len(imgField)
        height = len(imgField[0])

        centerScores = []
        for x in range(width):
            column = []
            for y in range(height):
                centerScore = self.scoreCenter(imgField, x, y)
                column.append(centerScore)

            centerScores.append(column)

        biggestCenter = None
        centerPos = None
        for x in range(width):
            for y in range(height):
                if biggestCenter is None:
                    biggestCenter = [x, y]
                    centerPos = (x, y)
                    continue

                xC = biggestCenter[0]
                yC = biggestCenter[1]

                if centerScores[x][y] > centerScores[xC][yC]:
                    biggestCenter = [x, y]
                    centerPos = (x, y)

        possibleCenters = [ str(biggestCenter[0]) + ":" + str(biggestCenter[1]) ]
        for x in range(width):
            for y in range(height):
                if x != centerPos[0] or y != centerPos[1]:
                    bigX = biggestCenter[0]
                    bigY = biggestCenter[1]
                    if abs(centerScores[bigX][bigY] - centerScores[x][y]) < 10:
                        center = str(x) + ":" + str(y)
                        possibleCenters.append(center)

        return possibleCenters

    # use scoring to get list of centers, and then have shade learning machine 'learn'
    # using each center as the list of the sample
    def scoreCenter(self, imgField, x, y):
        n = 0
        width = len(imgField)
        height = len(imgField[0])

        val = imgField[x][y]
        wVal = 0
        if x - 1 >= 0:
            wVal += 255 - imgField[x - 1][y]
            n += 1

        nVal = 0
        if y - 1 >= 0:
            nVal += 255 - imgField[x][y - 1]
            n += 1

        eVal = 0
        if x + 1 < width:
            eVal += 255 - imgField[x + 1][y]
            n += 1

        sVal = 0
        if y + 1 < height:
            sVal += 255 - imgField[x][y + 1]
            n += 1

        avgDiff = (wVal + nVal + eVal + sVal) / n

        score = imgField[x][y] - avgDiff

        return score

    def extractShadeData(self, data, centers, imgField):
        shades = []

        for center in centers:
            pos = center.split(":")
            x = int(pos[0])
            y = int(pos[1])

            shade = imgField[x][y]
            shades.append(shade)

        minShade = min(shades)
        maxShade = max(shades)

        data["avg"].extend(shades)
        data["min"].append(minShade)
        data["max"].append(maxShade)
