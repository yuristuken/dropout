import numpy
import scipy.special


class ActivationFunction:
    def __init__(self):
        pass

    def compute(self, value):
        pass

    def derivative(self, value):
        pass


class SigmoidActivationFunction(ActivationFunction):
    def compute(self, value):
        return scipy.special.expit(value)

    def derivative(self, value):
        return numpy.multiply(self.compute(value), 1 - self.compute(value))


class TanhActivationFunction(ActivationFunction):
    def compute(self, value):
        return numpy.tanh(value)

    def derivative(self, value):
        return 1 - numpy.power(value, 2)