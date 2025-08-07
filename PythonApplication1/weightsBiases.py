import numpy as np
import copy


class Layer_Dense:
    LayerDenseInstances = []
    whichDense = None

    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 *  np.random.randn(n_inputs, n_neurons )
        self.biases = np.zeros((1, n_neurons))
        self.neurons = n_neurons

        Layer_Dense.LayerDenseInstances.append(self)

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases 
