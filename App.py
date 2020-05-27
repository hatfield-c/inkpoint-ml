import random
import util.ImageSpace
import util.ImageFactory
from Matrix import Matrix

from TrainingFactory import TrainingFactory
from CenterLearning import CenterLearning
from ShadeLearning import ShadeLearning
from FeatureLearning import FeatureLearning
from Artist import Artist

import os
os.chdir("C:\\Users\\Cody\\Documents\\Professional\\inkpoint-ml\\")

class App:
    def __init__(self):
        #training = TrainingFactory()
        #training.buildTraining()
        print("Extracting edges...")
        util.ImageFactory.ImageFactory.ExtractEdges(inPath = "imgs\\personal\\me.jpg", outPath = "imgs\\personal\\me-edges.jpg")
        #return
        #agent = CenterLearning()
        #agent.learn()

        agent = ShadeLearning()
        #agent.learn(save = True)
        agent.load()

        #agent = FeatureLearning(imgPath = "imgs/sample_tiny.jpg")
        #agent.learn()

        artist = Artist(
            pen = agent,
            edgeSource = 'imgs\\personal\\me-edges.jpg',
            shadeSource = 'imgs\\personal\\me.jpg',
            resultPath = 'render\\draw-tests\\personal\\me.jpg'
        )
        artist.draw()

        '''a = agent.gen(index = 0, baseShade = agent.centerShade())
        a1 = agent.gen(index = 0, baseShade = agent.centerShade())
        a2 = agent.gen(index = 0, baseShade = agent.centerShade())
        a3 = agent.gen(index = 0, baseShade = agent.centerShade())
        a4 = agent.gen(index = 0, baseShade = agent.centerShade())
        a5 = agent.gen(index = 0, baseShade = agent.centerShade())
        b = agent.gen(1)
        c = agent.gen(2003)
        d = agent.gen(2005)
        e = agent.gen(4789)
        f = agent.gen(3577)
        g = agent.gen(-5)

        basePath = "C:/Users/Cody/Documents/Professional/inkpoint-ml/render/point-tests/"
        Matrix.SaveImage(matrix = a, path = basePath + "out0.png")
        Matrix.SaveImage(matrix = a1, path = basePath + "out0-1.png")
        Matrix.SaveImage(matrix = a2, path = basePath + "out0-2.png")
        Matrix.SaveImage(matrix = a3, path = basePath + "out0-3.png")
        Matrix.SaveImage(matrix = a4, path = basePath + "out0-4.png")
        Matrix.SaveImage(matrix = a5, path = basePath + "out0-5.png")
        Matrix.SaveImage(matrix = b, path = basePath + "out1.png")
        Matrix.SaveImage(matrix = c, path = basePath + "out2003.png")
        Matrix.SaveImage(matrix = d, path = basePath + "out2005.png")
        Matrix.SaveImage(matrix = e, path = basePath + "out4789.png")
        Matrix.SaveImage(matrix = f, path = basePath + "out3577.png")
        Matrix.SaveImage(matrix = g, path = basePath + "out-5.png")'''
        
    def seedValue(self):
        shade = 0
        while shade < 200 or shade > 255:
            shade = random.normalvariate(mu = 250, sigma = 13.75)

        return shade

app = App()
