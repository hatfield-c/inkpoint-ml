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
        #agent.learn()
        print(agent.pixelOrder())
        print(agent.pixelOrder())
        print(agent.pixelOrder())

app = App()
