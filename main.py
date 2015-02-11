from neural_network import *
from activation_functions import *

import random

if __name__ == "__main__":
    nn = NeuralNetwork([2, 5, 1], [TanhActivationFunction()] + [SigmoidActivationFunction()])

    #print nn.activate(numpy.array([[1, 2], [2, 3], [3, 4]]))

    #nn.back_propagation(numpy.array([[1, 2], [2, 3], [3, 4]]), numpy.array([0.33, 0.2, 0.14]), 0.1)
    #print nn.activate(numpy.array([[1, 2]]))[-1]
    #nn.print_weights()
    inputs = []
    targets =[]
    for i in range(1000):
        inputs.append([random.uniform(1, 10), random.uniform(1, 10)])
        targets.append([1.0/(inputs[-1][0] + inputs[-1][1])])

    inputs = numpy.array(inputs)
    targets = numpy.array(targets)
    for i in range(100000):
        nn.back_propagation(inputs, targets, 0.0001)
        if i % 10000 == 0:
            print i
            print nn.activate(numpy.array([[1, 2]]))[-1]

    nn.save_weights('weights.txt')

    #for i in range(100000):
        #nn.back_propagation(numpy.array([[1, 2], [2, 3]]), numpy.array([[0.33]]), 0.01)
    #    nn.back_propagation(numpy.array([[1, 2], [2, 3]]), numpy.array([[0.33], [0.2]]), 0.01)
    #    if i % 10000 == 0:
    #        print i
    #        print nn.activate(numpy.array([[1, 2]]))[-1]
    #        print nn.activate(numpy.array([[2, 3]]))[-1]

    nn.print_weights()
    #print nn.activate(numpy.array([[1, 2]]))[-1]
    #print nn.activate(numpy.array([[2, 3]]))[-1]