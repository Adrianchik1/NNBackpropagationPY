import numpy as np
import copy


class Layer_Dense:
    LayerDenseInstances = []
    totalWeights = []
    totalAmountOfWeights = 0
    totalAmountOfBiases = 0
    whichDense = None

    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 *  np.random.randn(n_inputs, n_neurons )
        self.biases = np.zeros((1, n_neurons))
        self.neurons = n_neurons
        Layer_Dense.totalWeights.append(self.weights)
        Layer_Dense.totalAmountOfWeights += len(self.weights) * len(self.weights[0])
        Layer_Dense.totalAmountOfBiases += len(self.weights) * len(self.weights[0])

        Layer_Dense.LayerDenseInstances.append(self)

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases 

    @classmethod
    def ChangeWeight(cls, denses, whichWeight, change):
        densesCopy = copy.deepcopy(denses)
        for dense in densesCopy:
            if whichWeight >= np.size(dense.weights):
                whichWeight -= np.size(dense.weights)
            else:
                row, col = np.unravel_index(whichWeight, dense.weights.shape)
                dense.weights[row, col] += change
                return densesCopy

    @classmethod
    def ChangeBias(cls, denses, whichBias, change): 
        densesCopy = copy.deepcopy(denses)
        for dense in densesCopy:
            if whichBias >= np.size(dense.biases):
                whichBias -= np.size(dense.biases)
            else:
                row, col = np.unravel_index(whichBias, dense.biases.shape)
                dense.biases[row, col] += change
                return densesCopy