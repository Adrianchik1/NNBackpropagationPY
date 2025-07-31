import numpy as np
import nnfs
from nnfs.datasets import spiral_data

from activations import Activation_ReLU, Activatioin_Softmax
from weightsBiases import Layer_Dense
from iterate import iteration               #importing needed functions from another files
from logicManager import Optimiser
from charts import makeChart

nnfs.init()
loss = float('inf')         #starting loss
iterations = 1000            #amount of iterations, how much times it would run the optimisation
change = 0.05               #amount by which weights and biases should change each iteration
losses = []                 #array to store the last 10 losses
differenceOfLosses = []     #array to store the differences of the last two losses(for future analysis)
implementBiases = False

dense1 = Layer_Dense(2, 10)         #weights and biases in the first layer
activation1 = Activation_ReLU()     #an activation class, that would multiply weights by inputs and would add biases

dense2 = Layer_Dense(10, 5)         #weights and biases in the second layer
activation2 = Activation_ReLU()     #another one activation class

dense3 = Layer_Dense(5, 3)          #weights and biases in the output layer
activation3 = Activatioin_Softmax() # 

X, y = spiral_data(samples=100, classes=3)

denses = [dense1, dense2, dense3]                       #all denses are added to one array, to pass them to another functions
activations = [activation1, activation2, activation3]   #all activations are added to one array, to pass them to another functions

optimiser = Optimiser(X, y, activations, change)        #creating an optimiser object, in future it would do the optimisation

for i in range(0, iterations):          #cycle which will optimize NN the required number of times
    print(f"Iteration {i}")
    loss, denses = optimiser.optimise(loss, denses)     #optimising denses
    print(f"Loss {loss}")

    losses.append(loss)                                                                     # appending the loss to the losses array
    lastTenLosses = np.array(losses[-10:])                                                  # getting the last 10 losses
    if np.allclose(lastTenLosses, lastTenLosses[0]) == True and len(losses) > 11:           # if loss hasnt changed in the last 10 iterations breaks
        break
    
    if len(losses) > 2 : differenceOfLosses.append(losses[i-1] - losses[i])                 # appends the change in loss to the progressOfLosses array


print(loss)
print(iterations)
makeChart(losses)
makeChart(differenceOfLosses)
input("Press enter to exit")