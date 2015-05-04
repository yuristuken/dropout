import numpy

class NeuralNetwork:
    #
    #    x
    #    x          y
    #    x          y          z
    #    x          y
    #    x
    #
    #    dim[0]     dim[1]     dim[2]
    #               act_f[0]   act_f[1]
    #    dp[0]      dp[1]
    #         w[0]       w[1]
    #
    #   Test output: act_f[1](act_f[0](input * (w[0] / (1-dp[0]))) * (w[1] / (1-dp[1])))
    #
    def __init__(self, dimensions, activation_functions, dropout_probabilities=[], max_norm=numpy.inf,):
        self.dimensions = dimensions
        self.activation_functions = activation_functions
        if (dropout_probabilities == []):
            self.dropout_probabilities = [0.0] * (len(dimensions) - 1)
        else:
            self.dropout_probabilities = dropout_probabilities
        # self.hidden_layers_count = len(dimensions)
        numpy.random.seed(45474547)
        self.weight_matrices = []
        self.velocities = []
        self.generate_random_weights()
        self.max_norm = max_norm
        #self.print_weights()
        return

    def generate_random_weights(self):
        self.weight_matrices = []
        for i in xrange(len(self.dimensions) - 1):
            self.weight_matrices.append(numpy.random.randn(self.dimensions[i] + 1, self.dimensions[i + 1]))
            self.velocities.append(numpy.zeros(self.weight_matrices[-1].shape))

    def load_weights(self, filename):
        matrices = numpy.load(filename)
        self.weight_matrices = [[]] * len(matrices.files)
        self.velocities = [[]] * len(matrices.files)
        for file_id in reversed(matrices.files):
            idx = int(file_id[4:])
            self.weight_matrices[idx] = matrices[file_id]
            self.velocities[idx] = numpy.zeros(self.weight_matrices[idx].shape)

    def save_weights(self, filename):
        f = open(filename, "w")
        numpy.savez(f, *self.weight_matrices)

    def print_weights(self):
        for i in xrange(len(self.dimensions) - 1):
            print "Weight matrix for " + str(i) + "->" + str(i + 1) + \
                  " (size: " + str(self.dimensions[i] + 1) + "*" + str(self.dimensions[i + 1]) + "): " + \
                  str(self.weight_matrices[i])

    def activate(self, input_values, test_mode=True):
        current_value = input_values
        activations = [current_value]

        for i, weights in enumerate(self.weight_matrices):
            current_value = self.append_bias_column(current_value)
            if test_mode:
                #weights_scaled = weights / (1 - self.dropout_probabilities[i])
                weights_scaled = weights * (1 - self.dropout_probabilities[i])
                current_value = self.activation_functions[i].compute(current_value.dot(weights_scaled))
                #current_value = self.activation_functions[i].compute(
                #    current_value.dot(weights / (1 - self.dropout_probabilities[i]))
                #)
            else:
                dropout_mask = (numpy.random.rand(*current_value.shape) > self.dropout_probabilities[i]).astype('float32')
                current_value = numpy.multiply(dropout_mask, current_value)
                # hacky, replace input
                if len(activations) == 1:
                    activations = [current_value[:, :-1]]
                current_value = self.activation_functions[i].compute(current_value.dot(weights))

            activations.append(current_value)
            #print "Layer " + str(i + 1) + ", Activation values: " + str(current_value)

        return activations

    def back_propagation(self, input_values, targets, learning_rate, momentum_coefficient=0.0, l2_regularization_coefficient=0.0):
        activations = self.activate(input_values, test_mode=False)

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
            self.velocities[i] = momentum_coefficient * self.velocities[i] - \
                                 learning_rate * numpy.dot(level_up_activations.T, deltas[-1])

            weight_decay = (1.0 - l2_regularization_coefficient * learning_rate) * numpy.ones((self.weight_matrices[i].shape[0] - 1, self.weight_matrices[i].shape[1]))
            weight_decay = numpy.concatenate((weight_decay, numpy.ones((1, self.weight_matrices[i].shape[1]))))
            #print weight_decay

            self.weight_matrices[i] = numpy.multiply(weight_decay, self.weight_matrices[i]) + self.velocities[i]

            # constrain max norm only for hidden units
            if i < len(self.dimensions) - 1:
                incoming_norms = numpy.apply_along_axis(numpy.linalg.norm, 0, self.weight_matrices[i])#numpy.linalg.norm(self.weight_matrices[i], axis=0)
                divisors = numpy.where(incoming_norms > self.max_norm, incoming_norms / self.max_norm, 1)
                self.weight_matrices[i] = self.weight_matrices[i] / divisors

            deltas.append(next_delta)

        return

    def append_bias_column(self, value):
        bias_column = numpy.ones((value.shape[0], 1))
        return numpy.concatenate((value, bias_column), axis=1)

