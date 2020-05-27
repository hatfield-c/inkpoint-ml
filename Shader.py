import math

class Shader:
    def __init__(self, shadeField, pointField):
        self.shadeField = shadeField
        self.pointField = pointField

    # *
    # * Determines the number of groups to sort a set of pixels into. The number of groups
    # * is determined by the range of potential pixel values in the set, and the maximum
    # * number of groups allowed.
    # *
    # * When the range of allowed values is 255 (the full pixel range), then the maximum
    # * number of groups will be returned. When the range of pixels enters a critical threshold,
    # * Then "1" will always be returned.
    # *
    def getNumGroups(self, k, valRange):
        kHalf = k / 2
        numGroups = k * (255 / (510 - valRange))
        numGroups -= (kHalf - 1) / (1 + math.exp((valRange - 75) / kHalf))
        numGroups = round(numGroups)

        return numGroups

                               