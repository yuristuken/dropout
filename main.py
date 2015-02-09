import scipy.special

from neural_network import *

if __name__ == "__main__":
    nn = NeuralNetwork([2, 10, 15, 2, 1], 3*[numpy.tanh] + [scipy.special.expit])

    nn.activate(numpy.array([[1, 2], [2, 3], [3, 4]]))
