import cv2
import random
from matplotlib import pyplot
from PIL import Image

from Matrix import Matrix
from util.ImageSpace import ImageSpace
from util.shader.Group import Group

class Artist:
    def __init__(self, pen, edgeSource, shadeSource, resultPath):
        self.pen = pen
        self.pointMin = 4
        self.lastPoint = 0

        self.edgeSource = edgeSource
        self.shadeSource = shadeSource
        self.resultPath = resultPath

        print("Opening edge source...")
        img = Image.open(self.edgeSource)
        imgSpace = ImageSpace(img = img)
        self.edgeField = imgSpace.getField(invert = True)

        print("Opening shade source...")
        img = Image.open(self.shadeSource)
        imgSpace = ImageSpace(img = img)
        self.shadeField = imgSpace.getField()

        self.width = len(self.shadeField)
        self.height = len(self.shadeField[0])

    def draw(self):
        print("Generating point/result fields...")
        traceField = Matrix.getEmptyMatrix(width = self.width, height = self.height)
        shadingField = Matrix.getEmptyMatrix(width = self.width, height = self.height)
        resultField = Matrix.getEmptyMatrix(width = self.width, height = self.height)
        
        print("Tracing edges...")
        self.trace(traceField)
        #print("Shading...")
        #self.shade(shadingField)
        print("Drawing image...")
        self.apply(resultField, traceField)
        #self.apply(resultField, shadingField)
        
        print("Saving...")
        Matrix.SaveImage(matrix = resultField, path = self.resultPath)

    def trace(self, pointField):
        for x in range(self.width):
            for y in range(self.height):
                if self.edgeField[x][y] < 70 and self.canPlace(pointField, x, y):
                    drawPoints = self.displace(x, y)
                    xPlace = drawPoints[0]
                    yPlace = drawPoints[1]

                    ImageSpace.SetPixel(pointField, xPlace, yPlace, 1)

    def shade(self, pointField):

        #groupField = X.getGroupField()

        return

    def apply(self, resultField, pointField):
        seedRange = 50000
        seedStep = 1
        seed = random.uniform(-seedRange, seedRange)

        for x in range(self.width):
            for y in range(self.height):
                if pointField[x][y] == 1:
                    point = self.pen.gen(index = seed, baseShade = self.shadeField[x][y])#self.pen.centerShade())

                    center = int((len(point) - 1) / 2)
                    
                    ImageSpace.writeSubImage(
                        image = resultField, 
                        sub = point, 
                        x = x, 
                        y = y, 
                        displace = center,
                        multiply = True
                    )

                    seed += seedStep
                    if seed > seedRange:
                        seed = random.uniform(-seedRange, seedRange)

    def canPlace(self, pointField, x, y):

        minR = 0
        for r in range(0, self.pointMin + 1):
            for i in range(x - r, x + r + 1):
                for j in range(y - r, y + r + 1):
                    if ImageSpace.SelectPixel(pointField, i, j) == 1:
                        return False

        return True


    def displace(self, x, y):
        probs = [[0.03, 0.09, 0.03],[0.09, 0.52, 0.09],[0.03, 0.09, 0.03]]

        width = len(probs)
        height = len(probs[0])
        center = int((width - 1) / 2)

        roll = random.random()
        cumulative = 0
        for i in range(width):
            for j in range(height):
                if roll < probs[i][j] + cumulative:
                    return [x + i - center, y + j - center]
                else:
                    cumulative += probs[i][j]

        # This should never happen.
        return None
        