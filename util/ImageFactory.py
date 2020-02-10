import cv2
from PIL import Image

class ImageFactory:

    @staticmethod
    def ExtractEdges(inPath, outPath, alpha = 100, beta = 200, inputType = 0):
        img = cv2.imread(inPath, inputType)
        edges = cv2.Canny(img, alpha, beta)
        cv2.imwrite(outPath, edges)