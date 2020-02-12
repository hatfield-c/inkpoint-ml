import math

class Shader:
    def __init__(self, shadeField, pointField):
        self.shadeField = shadeField
        self.pointField = pointField

    def getNumGroups(self, k, valRange):
        kHalf = k / 2
        numGroups = k * (255 / (510 - valRange))
        numGroups -= (kHalf - 1) / (1 + math.exp((valRange - 75) / kHalf))
        numGroups = round(numGroups)

        return numGroups

