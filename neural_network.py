import numpy

class NeuralNetwork:
    def __init__(self, dimensions, activation_functions):
        self.dimensions = dimensions
        self.activation_functions = activation_functions
        # self.hidden_layers_count = len(dimensions)
        self.weight_matrices = self.generate_random_weights()
        return

    def generate_random_weights(self):
        matrices = []
        for i in xrange(len(self.dimensions) - 1):
            matrices.append(numpy.random.uniform(0.0, 1.0, (self.dimensions[i] + 1, self.dimensions[i+1])))
            print "Weight matrix for " + str(i) + "->" + str(i+1) + \
                  " (size: " + str(self.dimensions[i] + 1) + "*" + str(self.dimensions[i+1]) + "): " + str(matrices[i])
        return matrices

    def activate(self, input):
        current_value = input

        for i, weights in enumerate(self.weight_matrices):
            # add bias to the input
            bias_column = numpy.ones((current_value.shape[0], 1))
            current_value = numpy.concatenate((current_value, bias_column), axis=1)
            current_value = self.activation_functions[i](current_value.dot(weights))
            print "Layer " + str(i + 1) + ", Activation values: " + str(current_value)