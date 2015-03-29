from neural_network import *
from activation_functions import *
from matplotlib import pyplot
from pylab import imshow, show, cm

import random
import logging



import os, struct
from array import array as pyarray
from numpy import append, array, int8, float_, zeros

def load_mnist(dataset="training", digits=numpy.arange(10), path="."):
    """
    Loads MNIST files into 3D numpy arrays

    Adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
    """

    if dataset == "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = len(ind)
    #N = 10

    images = zeros((N, rows*cols), dtype=float_)
    targets = zeros((N, 10), dtype=int8)
    for i in xrange(N):
        if i % 1000 == 0:
            logging.info("Loaded " + str(i) + " MNIST images")
        image = array(img[ind[i]*rows*cols : (ind[i]+1)*rows*cols])

        image_f = zeros(rows*cols, dtype=float_)

        for k in xrange(rows*cols):
            image_f[k] = image[k] / 255.0

        images[i] = image_f

        targets[i][lbl[ind[i]]] = 1
        #labels[i] = lbl[ind[i]]

    return images, targets


def load_array(filename):
    return numpy.load(filename)

def save_array(filename, x):
    f = open(filename, "w")
    numpy.save(f, x)

def convert_and_dump_MNIST():
    logging.info("Loading MNIST training dataset...")
    training_inputs, training_targets = load_mnist(dataset="training", path="/Users/yuristuken/PycharmProjects/dropout")
    logging.info("MNIST training dataset Loaded")

    logging.info("Dumping MNIST training dataset into file")
    save_array("training_inputs", training_inputs)
    save_array("training_targets", training_targets)
    logging.info("Training MNIST dataset dumped into file")

    logging.info("Loading MNIST testing dataset...")
    test_inputs, test_targets = load_mnist(dataset="testing", path="/Users/yuristuken/PycharmProjects/dropout")
    logging.info("MNIST testing dataset Loaded")

    logging.info("Dumping MNIST testing dataset into file")
    save_array("test_inputs", test_inputs)
    save_array("test_targets", test_targets)
    logging.info("Testing MNIST dataset dumped into file")

def load_mnist():
    logging.info("Loading MNIST")
    training_inputs = load_array("training_inputs")
    training_targets = load_array("training_targets")

    test_inputs = load_array("test_inputs")
    test_targets = load_array("test_targets")
    logging.info("MNIST loaded")

    return training_inputs, training_targets, test_inputs, test_targets

def generate_artificial_input():
    training_inputs = []
    training_targets =[]
    for i in range(1000):
        training_inputs.append([random.uniform(0, 1), random.uniform(0, 1)])
        training_targets.append([training_inputs[-1][0]**2 + training_inputs[-1][1]**3])

    training_inputs = numpy.array(training_inputs)
    training_targets = numpy.array(training_targets)

    test_inputs = []
    test_targets = []
    for i in range(500):
        test_inputs.append([random.uniform(0, 1), random.uniform(0, 1)])
        test_targets.append([test_inputs[-1][0]**2 + test_inputs[-1][1]**3])

    test_inputs = numpy.array(test_inputs)
    test_targets = numpy.array(test_targets)

    return training_inputs, training_targets, test_inputs, test_targets

def view_mnist_image(image):
    image = image.reshape(28,28).tolist()
    imshow(image, cmap=cm.gray)
    show()

if __name__ == "__main__":
    nn = NeuralNetwork([28*28, 10, 20, 10], [TanhActivationFunction()]*2 + [SigmoidActivationFunction()])

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    training_inputs, training_targets, test_inputs, test_targets = load_mnist()

    nn.load_weights('weights.txt')

    training_activations = nn.activate(training_inputs)[-1]
    training_error = numpy.mean((training_activations - training_targets) ** 2)
    test_activations = nn.activate(test_inputs)[-1]
    test_error = numpy.mean((test_activations - test_targets) ** 2)

    print training_error, test_error

    activations = nn.activate(test_inputs[1:10])[-1]
    for i in xrange(len(activations)):
        print activations[i]
        print activations[i].tolist().index(max(activations[i]))
        view_mnist_image(test_inputs[i])

    #print nn.activate(test_inputs[1:3])[-1]

    exit()

    epochs = []
    training_errors = []
    test_errors = []

    for i in xrange(300):
        sample_indices = random.sample(xrange(len(training_inputs)), 200)
        inputs_sample = numpy.array([training_inputs[k] for k in sample_indices])
        targets_sample = numpy.array([training_targets[k] for k in sample_indices])

        nn.back_propagation(inputs_sample, targets_sample, 0.0001)


        if i % 1 == 0:
            if i % 10 == 0:
                logging.info("Epoch: ", i, " Training error: ", training_error, " Test error: ", test_error)
            training_activations = nn.activate(training_inputs)[-1]
            training_error = numpy.mean((training_activations - training_targets) ** 2)
            test_activations = nn.activate(test_inputs)[-1]
            test_error = numpy.mean((test_activations - test_targets) ** 2)
            epochs.append(i)
            training_errors.append(training_error)
            test_errors.append(test_error)

    nn.save_weights('weights.txt')

    #print training_errors
    #print test_errors

    pyplot.plot(epochs, training_errors, 'b')
    pyplot.plot(epochs, test_errors, 'r')
    pyplot.show()

    #for i in range(100000):
        #nn.back_propagation(numpy.array([[1, 2], [2, 3]]), numpy.array([[0.33]]), 0.01)
    #    nn.back_propagation(numpy.array([[1, 2], [2, 3]]), numpy.array([[0.33], [0.2]]), 0.01)
    #    if i % 10000 == 0:
    #        print i
    #        print nn.activate(numpy.array([[1, 2]]))[-1]
    #        print nn.activate(numpy.array([[2, 3]]))[-1]

    #nn.print_weights()
    #print nn.activate(numpy.array([[1, 2]]))[-1]
    #print nn.activate(numpy.array([[2, 3]]))[-1]