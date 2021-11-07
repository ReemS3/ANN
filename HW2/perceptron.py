import numpy as np
from utils import sigmoid


class Perceptron():
    def __init__(self, number_of_input_units):
        self.bias = np.random.randn()
        self.weights = [np.random.randn()
                        for input_unit in range(number_of_input_units)]
        self.alpha = 1

    def forward_step(self, inputs):
        sum = self.bias

        for weight, input in zip(self.weights, inputs):
            sum += weight*input
        return sigmoid(sum)

    def update(self, deltas):
        # deltas consists of all deltas of the weights, plus the delta of bias. This is why we update the bias,
        # outside of the for-loop, since deltas' length is larger than self.weights by one
        updated_weights_tmp = []

        for index, weight in enumerate(self.weights):
            updated_weight = weight - self.alpha*deltas[index]
            updated_weights_tmp.append(updated_weight)
        self.weights = updated_weights_tmp

        self.bias = self.bias - self.alpha * deltas[-1]
