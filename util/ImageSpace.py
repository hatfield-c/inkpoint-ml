import math
from PIL import Image
from decimal import *

class ImageSpace:
    def __init__(self, img = None):
        self.img = img
        self.field = None

    def getField(self, invert = True):
        field = []

        width, height = self.img.size

        for x in range(width):
            column = []
            for y in range(height):
                pixel = self.img.getpixel((x, y))

                # Since the image (should) be black and white, which rgb
                # channel we choose from pixel[] shouldn't matter. Thus we
                # take the first channel (red).

                # Also, invert each pixel (which inverts the image) so that
                # weighting algorithms used later are simplified, i.e. black
                # pixels now have a shade value of 255, and white pixels have
                # a shade value of 0.
                # The resulting image will be de-inverted upon save.
                if isinstance(pixel, list) or isinstance(pixel, tuple):
                    if invert:
                        pixel = abs(255 - pixel[0])
                    else:
                        pixel = pixel[0]
                else:
                    if invert:
                        pixel = abs(255 - pixel)
                    else:
                        pixel = pixel

                column.append(pixel)

            field.append(column)

        return field

    def save(self, path, ftype):
        if path is None or ftype is None:
            print ("Invalid path or ftype!\npath: " + str(path) + "\nftype: " + str(ftype))
            return
        
        if self.img is None:
            print("No image to save!")
            return

        self.img.save(path, ftype)

    def show(self):
        self.img.show()

    @staticmethod
    def writeSubImage(image, sub, x, y, displace = 0, multiply = False):
        sWidth = len(sub)
        sHeight = len(sub[0])
        
        subVal = 0
        imgVal = 0
        for i in range(sWidth):
            for j in range(sHeight):
                xWrite = x + i - displace
                yWrite = y + j - displace
                
                if multiply:
                    subVal = 255 - sub[i][j]
                    imgVal = 255 - ImageSpace.SelectPixel(image, xWrite, yWrite)
                    
                    newVal = (imgVal / 255) * (subVal / 255)
                    newVal = int(255 * newVal)
                    newVal = 255 - newVal
                    
                else:
                    newVal = sub[i][j]

                ImageSpace.SetPixel(image, xWrite, yWrite, newVal)

    @staticmethod
    def ExtractNeighbors(x, y, imgField):
        shades = {}

        shades["n"] = ImageSpace.SelectPixel(
            imgField = imgField, 
            x = x, 
            y = y - 1
        )
        shades["ne"] = ImageSpace.SelectPixel(
            imgField = imgField, 
            x = x + 1, 
            y = y - 1
        )
        shades["e"] = ImageSpace.SelectPixel(
            imgField = imgField, 
            x = x + 1, 
            y = y
        )
        shades["se"] = ImageSpace.SelectPixel(
            imgField = imgField, 
            x = x + 1, 
            y = y + 1
        )
        shades["s"] = ImageSpace.SelectPixel(
            imgField = imgField, 
            x = x, 
            y = y + 1
        )
        shades["sw"] = ImageSpace.SelectPixel(
            imgField = imgField, 
            x = x - 1, 
            y = y + 1
        )
        shades["w"] = ImageSpace.SelectPixel(
            imgField = imgField, 
            x = x - 1, 
            y = y
        )
        shades["nw"] = ImageSpace.SelectPixel(
            imgField = imgField, 
            x = x - 1, 
            y = y - 1
        )

        return shades

    @staticmethod
    def SelectPixel(imgField, x, y):
        width = len(imgField)
        height = len(imgField[0])

        if x < 0:
            x = 0

        if x > width - 1:
            x = width - 1

        if y < 0:
            y = 0

        if y > height - 1:
            y = height - 1

        return imgField[x][y]

    @staticmethod
    def SetPixel(imgField, x, y, val):
        width = len(imgField)
        height = len(imgField[0])

        if x < 0:
            return

        if x > width - 1:
            return

        if y < 0:
            return

        if y > height - 1:
            return

        imgField[x][y] = val

    @staticmethod
    def BuildHistogram(image):
        width = len(image)
        height = len(image[0])
        h = { -1: { "num": 0, "numRange": 0 }}

        for i in range(256):
            h[i] = { "num": 0, "numRange": 0 }

        for x in range(width):
            for y in range(height):
                val = 255 - image[x][y]
                h[val]["num"] += 1

        for i in range(256):
                h[i]["numRange"] = h[i]["num"] + h[i - 1]["numRange"]

        return h

    @staticmethod 
    def EqualizeImage(image):
        histogram = ImageSpace.BuildHistogram(image)

        width = len(image)
        height = len(image[0])

        numPixels = width * height
        scaling = 256 / (numPixels * 2)

        equalized = {}
        for i in range(256):
            val = histogram[i]["numRange"] + histogram[i - 1]["numRange"]
            val = val * scaling

            if val >= 256:
                val = 255
            else:
                val = math.floor(val)

            equalized[i] = val

        for x in range(width):
            for y in range(height):
                imgVal = 255 - image[x][y]
                imgVal = equalized[imgVal]
                image[x][y] = 255 - imgVal