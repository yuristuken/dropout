import numpy

class NeuralNetwork:
    def __init__(self, dimensions, activation_functions):
        self.dimensions = dimensions
        self.activation_functions = activation_functions
        # self.hidden_layers_count = len(dimensions)
        #numpy.random.seed(1234)
        self.weight_matrices = []
        self.generate_random_weights()
        #self.print_weights()
        return

    def generate_random_weights(self):
        self.weight_matrices = []
        for i in xrange(len(self.dimensions) - 1):
            self.weight_matrices.append(numpy.random.randn(self.dimensions[i] + 1, self.dimensions[i + 1]))

    def load_weights(self, filename):
        matrices = numpy.load(filename)
        self.weight_matrices = [[]] * len(matrices.files)
        for file_id in reversed(matrices.files):
            idx = int(file_id[4:])
            self.weight_matrices[idx] = matrices[file_id]

    def save_weights(self, filename):
        f = open(filename, "w")
        numpy.savez(f, *self.weight_matrices)


    def print_weights(self):
        for i in xrange(len(self.dimensions) - 1):
            print "Weight matrix for " + str(i) + "->" + str(i + 1) + \
                  " (size: " + str(self.dimensions[i] + 1) + "*" + str(self.dimensions[i + 1]) + "): " + \
                  str(self.weight_matrices[i])

    def activate(self, input_values):
        current_value = input_values
        activations = [current_value]

        for i, weights in enumerate(self.weight_matrices):
            current_value = self.append_bias_column(current_value)
            current_value = self.activation_functions[i].compute(current_value.dot(weights))
            activations.append(current_value)
            #print "Layer " + str(i + 1) + ", Activation values: " + str(current_value)

        return activations

    def back_propagation(self, input_values, targets, learning_rate):
        activations = self.activate(input_values)

        deltas = [ numpy.multiply(self.activation_functions[-1].derivative(activations[-1]), activations[-1] - targets) ]

        # Iterate through layers backwards
        for i in reversed(xrange(len(self.dimensions) - 1)):
            #print level_up_activations.shape
            #print deltas[-1].shape
            #print learning_rate * numpy.dot(level_up_activations.T, deltas[-1])

            current_weights = self.weight_matrices[i][:-1, :]

            zzz = numpy.dot(current_weights, deltas[-1].T)
            #print zzz
            #print zzz.shape
            #print self.activation_functions[i-1].derivative(activations[i])
            #print self.activation_functions[i-1].derivative(activations[i]).shape
            next_delta = numpy.multiply(zzz.T, self.activation_functions[i-1].derivative(activations[i]))
            #print next_delta

            level_up_activations = self.append_bias_column(activations[i])
            self.weight_matrices[i] -= learning_rate * numpy.dot(level_up_activations.T, deltas[-1])

            deltas.append(next_delta)

        return

    def append_bias_column(self, value):
        bias_column = numpy.ones((value.shape[0], 1))
        return numpy.concatenate((value, bias_column), axis=1)

