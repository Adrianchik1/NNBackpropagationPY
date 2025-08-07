import numpy as np
import copy
from iterate import iteration
from weightsBiases import Layer_Dense
from calculateZ import CalculateZ
from calculateA import CalculateA
from losses import MSELoss

class Optimiser():
    def __init__(self, X, y, activations, change):
        self.X = X
        self.y = y
        self.activations = activations
        self.change = change

    def compute_delta_l(self, W_next, delta_next, z_l, activation_derivative):
        propagated = W_next @ delta_next  # shape (n_l, 1)
        sigma_prime = activation_derivative(z_l)  # shape (n_l, 1)
        delta_l = propagated * sigma_prime  # element-wise multiplication
        return delta_l

    def gradient(self, delta_l, a_prev):
        return np.dot( a_prev, delta_l.T)/ len(self.X)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        s = self.sigmoid(z)
        return s * (1 - s)

    def optimise(self):
        numberOfLayers = len(Layer_Dense.LayerDenseInstances)
        deltas =  [None for _ in range(numberOfLayers)]
        deltas[-1] = (MSELoss.backward())  
        for i in range((numberOfLayers-1), 0, -1):
            delta = self.compute_delta_l(Layer_Dense.LayerDenseInstances[i].weights, deltas[i], CalculateZ.CalculateZInstances[0].z[i-1], self.sigmoid_derivative)
            deltas[i-1] = delta

        for i, dense in enumerate(Layer_Dense.LayerDenseInstances):
            dense.weights -= self.change * self.gradient(deltas[i], CalculateA.a[i])
            dense.biases -= self.change * deltas[i].T.mean(axis=0, keepdims=True)

