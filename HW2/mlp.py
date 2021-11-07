from perceptron import Perceptron
from utils import sigmoidprime
import numpy as np

class MLP():
    def __init__(self):
        self.hidden_layer = [Perceptron(2), Perceptron(2), Perceptron(2), Perceptron(2)]
        self.output_layer = Perceptron(4)
        self.previous_activations = []
        self.output = 0
        
    def backpropagate(self, target):
        error = self.output - target
        deltas = []
        error_output = sigmoidprime(self.output)*error      
        for index in range(len(self.hidden_layer)):

            deltas.append(self.previous_activations[index]* error)
        deltas.append(error_output)
        self.output_layer.update(deltas)

    
    def forwardpropagate(self, input):
        self.previous_activations = []
        for perpectron in self.hidden_layer:
            output = perpectron.forward_step(input)
            # self.deriviative_previous_activations.append(output)    
            self.previous_activations.append(output)
        previous_activations_reshaped = np.reshape(self.previous_activations, newshape=(-1))
        self.output = self.output_layer.forward_step(previous_activations_reshaped)
        # self.previous_activations.append(output)