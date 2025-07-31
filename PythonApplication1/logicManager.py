import numpy as np
import copy
from iterate import iteration
from weightsBiases import Layer_Dense

class Optimiser():
    def __init__(self, X, y, activations, change):
        self.X = X
        self.y = y
        self.activations = activations
        self.change = change

    def optimiseSingleWeight(self, loss, denses, nthElem):
        
        positiveDenses = Layer_Dense.ChangeWeight(denses, nthElem, self.change)     #change Nth weight by change
        negativeDenses = Layer_Dense.ChangeWeight(denses, nthElem, -self.change)

        positiveLoss = iteration(self.X, self.y, positiveDenses, self.activations)  #calculates new loss
        negativeLoss = iteration(self.X, self.y, negativeDenses, self.activations)
        loss = iteration(self.X, self.y, denses, self.activations)                  #calculates loss with old weights

        losses = [(positiveLoss, positiveDenses),
                  (negativeLoss, negativeDenses),
                  (loss, denses)] 
        return min(losses, key=lambda x: x[0])      #comperse losses, and outputs the best loss and denses

    def optimiseSingleBias(self, loss, denses, nthElem):
        positiveDenses = Layer_Dense.ChangeBias(denses, nthElem, self.change)
        negativeDenses = Layer_Dense.ChangeBias(denses, nthElem, -self.change)

        positiveLoss = iteration(self.X, self.y, positiveDenses, self.activations)
        negativeLoss = iteration(self.X, self.y, negativeDenses, self.activations)

        losses = [(positiveLoss, positiveDenses),
                  (negativeLoss, negativeDenses),
                  (loss, denses)] 
        return min(losses, key=lambda x: x[0]) 
     
     
    def calculateAmountOfWeights(self, denses):
        amountOfWeights = []
        for dense in denses:
            amountOfWeightsInDense = len(dense.weights.flatten())
            amountOfWeights.append(amountOfWeightsInDense)
        return amountOfWeights

    def whichBias(self, denses, amountOfWeights, whichWeight):
        whichBias = 0
        for i in range(len(denses)):
            if amountOfWeights[i] > whichWeight:
                whichBias += whichWeight % len(denses[i].biases[0])
                return whichBias
            else:
                whichWeight -= amountOfWeights[i]
                whichBias += len(denses[i].biases[0]) 


    def optimise(self, loss, denses):
        amountOfWeights = self.calculateAmountOfWeights(denses)             #calculates amount of weights in every dense(posible output: [35, 15, 9])
        for whichWeight in range(Layer_Dense.totalAmountOfWeights):         #changes one weight per iteration
            whichBias = self.whichBias(denses, amountOfWeights, whichWeight)        #calculates which bias should be changed
            loss, denses = self.optimiseSingleWeight(loss, denses, whichWeight)     #optimises single weight
            #whichBias = whichWeight//len(denses[0].biases[0])
            loss, denses = self.optimiseSingleBias(loss, denses, whichBias)

        return loss, denses

