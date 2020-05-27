from PIL import Image
from Matrix import Matrix
from util.ImageSpace import ImageSpace

from util.machine.LeastSquaresMultiOutput import LeastSquaresMultiOutput as ML

class FeatureLearning:

    # These do nothing - they simply exist as a helpful reference
    INPUTS = [
        "Pixels of the image"
        "pix0",
        "pix1",
        "...",
        "pixN",
    ]

    def __init__(self, imgPath, loadPath = None, savePath = None):
        self.loadPath = loadPath
        self.savePath = savePath

        self.theta = {
            "V": None,
            "W": None
        }

        print("Reading sample image...")
        self.imgPath = imgPath
        self.inputUnits = self.readSampleImage()

        self.sampleWidth = len(self.inputUnits)
        self.sampleHeight = len(self.inputUnits[0])

        print("Vectorizing image...")
        self.inputUnits = Matrix.pixNorm(self.inputUnits)
        self.inputUnits = Matrix.vec(self.inputUnits)

        self.machine = None

        self.nOutput = len(self.inputUnits[0])
        self.nHidden = 17
        self.nInputs = self.nOutput

        self.learnRate = 0.001

    def learn(self):
        if self.machine is None:
            print("Initializing machine...")
            self.machine = ML(
                nOutput = self.nOutput,
                nHidden = self.nHidden,
                nInputs = self.nInputs,
                learnRate = self.learnRate
            )

        print("Begin learning...")
        descend = True
        passCount = 1
        while(descend):

            resultData = self.machine.learnOnce(desired = self.inputUnits, inputData = self.inputUnits)
            groupPredictions = resultData["predicted"]
            cost = resultData["cost"]

            ln = cost

            print(" ------------ Pass: " + str(passCount) +  " ------------ ")
            print("ln   :", ln)

            passCount += 1
            if ln < 5:
                descend = False

    def readSampleImage(self):
        img = Image.open(self.imgPath)
        imgSpace = ImageSpace(img = img)

        return imgSpace.getField(invert = False)