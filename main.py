from neural_network import *
from activation_functions import *
from matplotlib import pyplot
from pylab import imshow, show, cm

import random
import logging
import pickle

import os, struct
from numpy import append, array, int8, float_, zeros


def load_mnist_from_binary(dataset="training", digits=numpy.arange(10), path="."):
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

    ind = [k for k in range(size) if lbl[k] in digits]
    N = len(ind)
    # N = 10

    images = zeros((N, rows * cols), dtype=float_)
    targets = zeros((N, 10), dtype=int8)
    for i in xrange(N):
        if i % 1000 == 0:
            logging.info("Loaded " + str(i) + " MNIST images")
        image = array(img[ind[i] * rows * cols: (ind[i] + 1) * rows * cols])

        image_f = zeros(rows * cols, dtype=float_)

        for k in xrange(rows * cols):
            image_f[k] = image[k] / 255.0

        images[i] = image_f

        targets[i][lbl[ind[i]]] = 1
        # labels[i] = lbl[ind[i]]

    return images, targets


def load_array(filename):
    return numpy.load(filename)


def save_array(filename, x):
    f = open(filename, "w")
    numpy.save(f, x)


def convert_and_dump_MNIST():
    logging.info("Loading MNIST training dataset...")
    training_inputs, training_targets = load_mnist_from_binary(dataset="training", path="/Users/yuristuken/PycharmProjects/dropout")
    logging.info("MNIST training dataset Loaded")

    logging.info("Dumping MNIST training dataset into file")
    save_array("training_inputs", training_inputs)
    save_array("training_targets", training_targets)
    logging.info("Training MNIST dataset dumped into file")

    logging.info("Loading MNIST testing dataset...")
    test_inputs, test_targets = load_mnist_from_binary(dataset="testing", path="/Users/yuristuken/PycharmProjects/dropout")
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
    training_targets = []
    for i in range(1000):
        training_inputs.append([random.uniform(0, 1), random.uniform(0, 1)])
        training_targets.append([training_inputs[-1][0] ** 2 + training_inputs[-1][1] ** 3])

    training_inputs = numpy.array(training_inputs)
    training_targets = numpy.array(training_targets)

    test_inputs = []
    test_targets = []
    for i in range(500):
        test_inputs.append([random.uniform(0, 1), random.uniform(0, 1)])
        test_targets.append([test_inputs[-1][0] ** 2 + test_inputs[-1][1] ** 3])

    test_inputs = numpy.array(test_inputs)
    test_targets = numpy.array(test_targets)

    return training_inputs, training_targets, test_inputs, test_targets


def view_mnist_image(image):
    image = image.reshape(28, 28).tolist()
    imshow(image, cmap=cm.gray)
    show()


def compute_mnist_correct_classifications(activations, targets):
    target_classes = [numpy.argmax(target) for target in targets]
    predicted_classes = [numpy.argmax(activation) for activation in activations]
    matches = 0
    for i in xrange(len(target_classes)):
        if target_classes[i] == predicted_classes[i]:
            matches += 1
    return matches


if __name__ == "__main__":
    nn = NeuralNetwork([28 * 28, 30, 10], [TanhActivationFunction()] + [SigmoidActivationFunction()])

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    training_inputs, training_targets, test_inputs, test_targets = load_mnist()

    epochs = []
    training_errors = []
    test_errors = []
    training_corrects = []
    test_corrects = []

    n_epochs = 30
    batch_size = 20
    learning_rate = 0.1

    logging.info(
        'Starting learning for {n_epochs} epochs with batch size of {batch_size} and learning rate of {learning_rate}'.format(
            n_epochs=n_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        ))

    for i in xrange(n_epochs):
        shuffled_indices = range(len(training_inputs))
        random.shuffle(shuffled_indices)

        mini_batch_indices = [shuffled_indices[k:k + batch_size] for k in xrange(0, len(training_inputs), batch_size)]

        for mini_batch in mini_batch_indices:
            inputs_sample = numpy.array([training_inputs[k] for k in mini_batch])
            targets_sample = numpy.array([training_targets[k] for k in mini_batch])

            nn.back_propagation(inputs_sample, targets_sample, learning_rate)

        training_activations = nn.activate(training_inputs)[-1]
        training_error = numpy.mean((training_activations - training_targets) ** 2)
        training_correct = compute_mnist_correct_classifications(training_activations, training_targets)
        test_activations = nn.activate(test_inputs)[-1]
        test_error = numpy.mean((test_activations - test_targets) ** 2)
        test_correct = compute_mnist_correct_classifications(test_activations, test_targets)

        epochs.append(i)
        training_errors.append(training_error)
        test_errors.append(test_error)
        training_corrects.append(training_correct)
        test_corrects.append(test_correct)

        if i % 1 == 0:
            logging.info(
                'Epoch: {epoch}, Training error: {training_error}, Test error: {test_error}, Training correct: {training_correct}, Test correct: {test_correct}'.format(
                    epoch=i,
                    training_error=training_error,
                    test_error=test_error,
                    training_correct=training_correct,
                    test_correct=test_correct
                ))

    nn.save_weights('results/weights')

    f = open('results/epochs', "w")
    pickle.dump(epochs, f)
    f.close()
    f = open('results/training_errors', "w")
    pickle.dump(training_errors, f)
    f.close()
    f = open('results/test_errors', "w")
    pickle.dump(test_errors, f)
    f.close()
    f = open('results/training_correct', "w")
    pickle.dump(training_correct, f)
    f.close()
    f = open('results/test_correct', "w")
    pickle.dump(test_correct, f)
    f.close()

    pyplot.plot(epochs, training_errors, 'b')
    pyplot.plot(epochs, test_errors, 'r')
    pyplot.show()

    pyplot.plot(epochs, [i / 60000.0 for i in training_corrects], 'b')
    pyplot.plot(epochs, [i / 10000.0 for i in test_corrects], 'r')
    pyplot.show()
