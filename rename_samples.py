import os
from PIL import Image

rootPath = "C:/Users/Cody/Documents/Professional/inkpoint-ml/samples"
files = os.listdir("C:/Users/Cody/Documents/Professional/inkpoint-ml/samples")

newName = "input"
index = 0
for fileName in files:
    # Make three copies of the image, and invert each one about a respective x/y axis. This creates four useful samples
    # from just one, as it is unlikely that a sample will be perfectly mirrored along an axis

    os.rename(r'' + rootPath + '/' + fileName, r'' + rootPath + '/' + newName + str(index) + '.png')

    normalImg = Image.open(rootPath + '/' + newName + str(index) + '.png')
    
    mirrorX = normalImg.transpose(Image.FLIP_LEFT_RIGHT)
    mirrorY = normalImg.transpose(Image.FLIP_TOP_BOTTOM)
    mirrorXY = mirrorY.transpose(Image.FLIP_LEFT_RIGHT)

    index += 1
    mirrorX.save(rootPath + '/' + newName + str(index) + '.png')
    index += 1
    mirrorY.save(rootPath + '/' + newName + str(index) + '.png')
    index += 1
    mirrorXY.save(rootPath + '/' + newName + str(index) + '.png')

    index += 1

