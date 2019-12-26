from PIL import Image
from decimal import *

class ImageSpace:
    def __init__(self, img = None):
        self.img = img
        self.field = None

    def getField(self):
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
                pixel = abs(255 - pixel[0])

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
