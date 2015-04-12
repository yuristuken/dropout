from neural_network import *
from activation_functions import *
from matplotlib import pyplot
from pylab import imshow, show, cm

import logging
import pickle

import os, struct
from numpy import append, array, int8, float_, zeros
from array import array as pyarray


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


def train_wrapper(**kwargs):
    nnargs = {'dimensions': kwargs['dimensions'], 'activation_functions': kwargs['activation_functions']}

    experiment_name = ','.join(str(d) for d in kwargs['dimensions'])
    experiment_name += '_activations=' + ','.join(a.__class__.__name__ for a in kwargs['activation_functions'])

    if 'dropout_probabilities' in kwargs:
        experiment_name += '_dropout=' + ','.join(str(p) for p in kwargs['dropout_probabilities'])
        nnargs['dropout_probabilities'] = kwargs['dropout_probabilities']
    else:
        experiment_name += '_nodropout'

    if 'max_norm' in kwargs:
        experiment_name += '_maxNorm=' + str(kwargs['max_norm'])
        nnargs['max_norm'] = kwargs['max_norm']

    experiment_name += '_epochs=' + str(kwargs['epochs'])
    experiment_name += '_learningRate=' + str(kwargs['learning_rate'])
    experiment_name += '_momentum=' + str(kwargs['momentum_coefficient'])
    experiment_name += '_regCoefficient=' + str(kwargs['l2_regularization_coefficient'])
    experiment_name += '_batchSize=' + str(kwargs['batch_size'])

    print "Experiment: " + experiment_name
    n_epochs = kwargs['epochs']
    batch_size = kwargs['batch_size']
    learning_rate = kwargs['learning_rate']
    momentum_coefficient = kwargs['momentum_coefficient']
    l2_regularization_coefficient = kwargs['l2_regularization_coefficient']

    nn = NeuralNetwork(**nnargs)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    training_inputs, training_targets, test_inputs, test_targets = load_mnist()

    #validation_inputs = training_inputs[1000:6000]
    #validation_targets = training_targets[1000:6000]
    validation_inputs = test_inputs
    validation_targets = test_targets

    rng_state = numpy.random.get_state()
    numpy.random.shuffle(training_inputs)
    numpy.random.set_state(rng_state)
    numpy.random.shuffle(training_targets)

    training_inputs = training_inputs[:60000]
    training_targets = training_targets[:60000]

    epochs = []
    training_errors = []
    #test_errors = []
    validation_errors = []
    training_corrects = []
    #test_corrects = []
    validation_corrects = []

    initial_learning_rate = learning_rate

    logging.info(
        'Starting learning for {n_epochs} epochs with batch size of {batch_size} and learning rate of {learning_rate}'.format(
            n_epochs=n_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        ))

    for i in xrange(n_epochs):
        shuffled_indices = range(len(training_inputs))
        numpy.random.shuffle(shuffled_indices)

        mini_batch_indices = [shuffled_indices[k:k + batch_size] for k in xrange(0, len(training_inputs), batch_size)]

        for mini_batch in mini_batch_indices:
            inputs_sample = numpy.array([training_inputs[k] for k in mini_batch])
            targets_sample = numpy.array([training_targets[k] for k in mini_batch])

            nn.back_propagation(inputs_sample, targets_sample, learning_rate, momentum_coefficient, l2_regularization_coefficient)

        learning_rate = initial_learning_rate / (1.0 + 0.01*float(i))
        print "new learning rate: " + str(learning_rate)

        training_activations = nn.activate(training_inputs)[-1]
        training_error = numpy.mean((training_activations - training_targets) ** 2)
        training_correct = compute_mnist_correct_classifications(training_activations, training_targets)

        #test_activations = nn.activate(test_inputs)[-1]
        #test_error = numpy.mean((test_activations - test_targets) ** 2)
        #test_correct = compute_mnist_correct_classifications(test_activations, test_targets)

        validation_activations = nn.activate(validation_inputs)[-1]
        validation_error = numpy.mean((validation_activations - validation_targets) ** 2)
        validation_correct = compute_mnist_correct_classifications(validation_activations, validation_targets)

        epochs.append(i)

        training_errors.append(training_error)
        training_corrects.append(training_correct)

        #test_errors.append(test_error)
        #test_corrects.append(test_correct)

        validation_errors.append(validation_error)
        validation_corrects.append(validation_correct)

        if i % 1 == 0:
            logging.info(
                'Epoch: {epoch}, '.format(epoch=i) +
                'Training error: {training_error}, Validation error: {validation_error}, '.format(
                    training_error=training_error,
                    validation_error=validation_error
                ) +
                'Training correct: {training_correct} ({training_correct_percent}%), '.format(
                    training_correct=training_correct,
                    training_correct_percent=int(float(training_correct)/len(training_inputs)*10000) / 100.0
                ) +
                'Validation correct: {validation_correct} ({validation_correct_percent}%)'.format(
                    validation_correct=validation_correct,
                    validation_correct_percent=int(float(validation_correct)/len(validation_inputs)*10000) / 100.0
                ))

    path = 'results_20150411_big/' + experiment_name

    if not os.path.exists(path):
        os.makedirs(path)

    nn.save_weights(path + '/weights')

    f = open(path + '/epochs', "w")
    pickle.dump(epochs, f)
    f.close()
    f = open(path + '/training_errors', "w")
    pickle.dump(training_errors, f)
    f.close()
    #f = open(path + '/test_errors', "w")
    #pickle.dump(test_errors, f)
    #f.close()
    f = open(path + '/validation_errors', "w")
    pickle.dump(validation_errors, f)
    f.close()
    f = open(path + '/training_corrects', "w")
    pickle.dump(training_corrects, f)
    f.close()
    f = open(path + '/validation_corrects', "w")
    pickle.dump(validation_corrects, f)
    f.close()

    pyplot.plot(epochs, training_errors, 'b')
    #pyplot.plot(epochs, test_errors, 'r')
    pyplot.plot(epochs, validation_errors, 'g')
    pyplot.savefig(path + '/errors.png')
    pyplot.figure()
    #pyplot.show()

    pyplot.plot(epochs, [float(i) / len(training_inputs) for i in training_corrects], 'b')
    #pyplot.plot(epochs, [float(i) / len(test_inputs) for i in test_corrects], 'r')
    pyplot.plot(epochs, [float(i) / len(validation_inputs) for i in validation_corrects], 'g')
    pyplot.savefig(path + '/corrects.png')
    pyplot.figure()


if __name__ == "__main__":
    for learning_rate in [0.0001]:
        for max_norm in [4.0, 6.0]:
            train_wrapper(epochs=150,
                          learning_rate=learning_rate,
                          batch_size=50,
                          dimensions=[28 * 28, 1024, 1024, 10],
                          activation_functions=[TanhActivationFunction(), TanhActivationFunction(), SigmoidActivationFunction()],
                          dropout_probabilities=[0.5, 0.5, 0.5],
                          max_norm=max_norm,
                          momentum_coefficient=0.95,
                          l2_regularization_coefficient=0.0
            )

