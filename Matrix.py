import random
import math
import csv

from PIL import Image

class Matrix:
    @staticmethod
    def addMatrix(a, b):
        width = len(a)
        height = len(a[0])

        result = Matrix.getEmptyMatrix(width, height)

        for x in range(width):
            for y in range(height):
                result[x][y] = a[x][y] + b[x][y]

        return result

    @staticmethod
    def subMatrix(a, b):
        width = len(a)
        height = len(a[0])

        result = Matrix.getEmptyMatrix(width, height)

        for x in range(width):
            for y in range(height):
                result[x][y] = a[x][y] - b[x][y]

        return result

    @staticmethod
    def multMatrix(a, b):
        width = len(a)
        height = len(a[0])

        result = Matrix.getEmptyMatrix(width, height)

        for x in range(width):
            for y in range(height):
                result[x][y] = a[x][y] * b[x][y]

        return result

    @staticmethod
    def divMatrix(a, b):
        width = len(a)
        height = len(a[0])

        result = Matrix.getEmptyMatrix(width, height)

        for x in range(width):
            for y in range(height):
                result[x][y] = a[x][y] / b[x][y]

        return result

    @staticmethod
    def linearTransform(left, right):
        leftWidth = len(left)
        rightHeight = len(right[0])

        width = len(right)
        height = len(left[0])
        result = Matrix.getEmptyMatrix(width, height)

        for x in range(width):
            for y in range(height):        
                entry = 0

                # Perform transformation, with i and j making sure that the
                # rows in the right matrix (j) match the columns in the left
                # matrix (i). If they don't, an index exception is thrown.
                i = 0
                j = 0
                while i < leftWidth or j < rightHeight:
                    entry += left[j][y] * right[x][i]
                    i += 1
                    j += 1
                
                result[x][y] = entry

        return result

    @staticmethod
    def transpose(orig):
        rows = []
        for y in range(len(orig[0])):
            row = []
            rows.append(row)

        for x in range(len(orig)):
            for y in range(len(orig[0])):
                rows[y].append(orig[x][y])

        # Due to how python represents pointers, a list of rows from 
        # a matrix - where a matrix is a list of columns - is functionally 
        # equivalent to a transpose operation on said matrix (which to
        # reiterate, is a list of columns being transposed into a list of
        # rows).
        return rows

    @staticmethod
    def diag(vector):
        length = len(vector[0])
        matrix = Matrix.getEmptyMatrix(length, length)

        for y in range(length):
            entry = vector[0][y]
            matrix[y][y] = entry

        return matrix

    @staticmethod
    def vec(matrix):
        vector = []
        for x in range(len(matrix)):
            for y in range(len(matrix[0])):
                entry = matrix[x][y]
                vector.append(entry)

        return [ vector ]

    @staticmethod
    def generateI(size):
        i = Matrix.getEmptyMatrix(size, size)

        for x in range(size):
            for y in range(size):
                i[x][y] = 1

        return i

    @staticmethod
    def columnOfI(i, size):
        column = []

        for x in range(size):
            if x == i:
                entry = 1
            else:
                entry = 0

            column.append(entry)

        return [ column ]

    @staticmethod
    def getEmptyMatrix(width, height, default = 0):
            empty = []

            for x in range(width):
                column = []
                for y in range(height):
                    weight = default

                    column.append(weight)

                empty.append(column)

            return empty

    @staticmethod
    def getOneHotMatrix(width, height, x, y):
        matrix = Matrix.getEmptyMatrix(width = width, height = height)
        matrix[x][y] = 1

        return Matrix

    @staticmethod
    def getRandomWeights(width, height):
            weights = []

            for x in range(width):
                column = []
                for y in range(height):
                    weight = random.uniform(-1, 1)

                    column.append(weight)

                weights.append(column)

            return weights

    @staticmethod
    def DIAG(vector):
        vector = vector[0]
        size = len(vector)

        result = Matrix.getEmptyMatrix(size, size)
        for x in range(size):
            for y in range(size):
                if x == y:
                    result[x][y] = vector[x]

        return result

    # Element by element application of sigmoidal function onto
    # elements of a matrix
    @staticmethod
    def sigmoidal(matrix):
        width = len(matrix)
        height = len(matrix[0])

        result = Matrix.getEmptyMatrix(width, height)

        for x in range(width):
            for y in range(height):
                try:
                    entry = matrix[x][y]
                    sigm = 1 / (1 + math.exp((-1) * entry))
                except:
                    print(entry)
                    #exit()

                result[x][y] = sigm

        return result

    @staticmethod
    def softplus(matrix):
        width = len(matrix)
        height = len(matrix[0])

        result = Matrix.getEmptyMatrix(width, height)

        for x in range(width):
            for y in range(height):
                entry = matrix[x][y]
                softp = math.log(1 + math.exp(entry))

                result[x][y] = softp

        return result

    @staticmethod
    def log(matrix):
        width = len(matrix)
        height = len(matrix[0])

        result = Matrix.getEmptyMatrix(width, height)

        for x in range(width):
            for y in range(height):
                result[x][y] = math.log(matrix[x][y])

        return result

    @staticmethod
    def prob(matrix):
        width = len(matrix)
        height = len(matrix[0])

        result = Matrix.getEmptyMatrix(width, height)

        for x in range(width):
            for y in range(height):
                result[x][y] = 1 / (1 + math.exp((-1) * matrix[x][y]))

        return result

    @staticmethod
    def printM(matrix):
        width = len(matrix)
        height = len(matrix[0])

        for y in range(height):
            line = "[ "
            for x in range(width):
                entry = matrix[x][y]
                line += str(entry) + " | "
            
            line = line[0:-3]
            line += " ]"
            print(line)
        
        print("")

    @staticmethod
    def multScalar(matrix, scalar):
        width = len(matrix)
        height = len(matrix[0])

        result = Matrix.getEmptyMatrix(width, height)
        for x in range(width):
            for y in range(height):
                result[x][y] = scalar * matrix[x][y]

        return result

    @staticmethod
    def toInteger(matrix):
        width = len(matrix)
        height = len(matrix[0])

        result = Matrix.getEmptyMatrix(width, height)
        for x in range(width):
            for y in range(height):
                result[x][y] = int(round(matrix[x][y]))
            
        return result

    @staticmethod
    def stringToFloat(matrix):
        width = len(matrix)
        height = len(matrix[0])

        result = Matrix.getEmptyMatrix(width, height)
        for x in range(width):
            for y in range(height):
                result[x][y] = float(matrix[x][y])
            
        return result
        

    @staticmethod
    def getRow(matrix, k):
        width = len(matrix)
        height = len(matrix[0])

        result = []
        for x in range(width):
            for y in range(height):
                if y == k:
                    entry = [ matrix[x][y] ]
                    break
                
            result.append(entry)

        return result

    @staticmethod
    def pixNorm(matrix):
        return Matrix.normEntries(matrix = matrix, maxVal = 255)

    @staticmethod
    def normEntries(matrix, maxVal):
        width = len(matrix)
        height = len(matrix[0])

        result = []

        for x in range(width):
            column = []
            for y in range(height):
                entry = matrix[x][y]
                entry = entry / maxVal
                column.append(entry)

            result.append(column)

        return result

    @staticmethod
    def invertNormData(matrix):
        width = len(matrix)
        height = len(matrix[0])

        result = []

        for x in range(width):
            column = []
            for y in range(height):
                entry = matrix[x][y]
                entry = 1 - entry
                column.append(entry)

            result.append(column)

        return result

    @staticmethod
    def SaveImage(matrix, path):
        width = len(matrix)
        height = len(matrix[0])

        img = Image.new(mode = "RGB", size = (width, height))
        pixels = img.load()

        for x in range(width):
            for y in range(height):
                entry = 255 - matrix[x][y]
                pixels[x, y] = (entry, entry, entry)

        img.save(path)

    @staticmethod
    def SaveCSV(matrix, path):
        width = len(matrix)
        height = len(matrix[0])

        matrixTrans = Matrix.transpose(orig = matrix)
        with open(path, mode = "w", newline = '') as matrixFile:
            writer = csv.writer(matrixFile)
            
            for column in matrixTrans:
                writer.writerow(column)

    @staticmethod
    def LoadCSV(path):

        rows = []

        with open(path, mode = "r", newline = '') as matrixFile:
            csvReader = csv.reader(matrixFile)

            for row in csvReader:
                rows.append(row)

        result = Matrix.transpose(rows)
        return result