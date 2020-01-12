import random
import util.ImageSpace
from Matrix import Matrix

from TrainingFactory import TrainingFactory
from CenterLearning import CenterLearning
from ShadeLearning import ShadeLearning

class App:
    def __init__(self):
        #training = TrainingFactory()
        #training.buildTraining()

        #agent = CenterLearning()
        #agent.learn()

        agent = ShadeLearning()
        agent.learn(save = True)

        a = agent.gen(0)
        b = agent.gen(1)
        c = agent.gen(2003)
        d = agent.gen(2005)
        e = agent.gen(4789)
        f = agent.gen(3577)
        g = agent.gen(-5)

        basePath = "C:/Users/Cody/Documents/Professional/inkpoint-ml/render/point-tests/"
        Matrix.SaveImage(matrix = a, path = basePath + "out0.png")
        Matrix.SaveImage(matrix = b, path = basePath + "out1.png")
        Matrix.SaveImage(matrix = c, path = basePath + "out2003.png")
        Matrix.SaveImage(matrix = d, path = basePath + "out2005.png")
        Matrix.SaveImage(matrix = e, path = basePath + "out4789.png")
        Matrix.SaveImage(matrix = f, path = basePath + "out3577.png")
        Matrix.SaveImage(matrix = g, path = basePath + "out-5.png")
        
    def seedValue(self):
        shade = 0
        while shade < 200 or shade > 255:
            shade = random.normalvariate(mu = 250, sigma = 13.75)

        return shade

app = App()
