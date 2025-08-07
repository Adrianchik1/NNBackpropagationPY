import numpy as np

from activations import Activation_ReLU, Activatioin_Softmax
from weightsBiases import Layer_Dense
from iterate import iteration               #importing needed functions from another files
from logicManager import Optimiser
from charts import makeChart
from losses import MSELoss
from generateData import generateData

loss = float('inf')         #starting loss
iterations = 100000            #amount of iterations, how much times it would run the optimisation
change = 0.05               #amount by which weights and biases should change each iteration
losses = []                 #array that stores the last 10 losses
differenceOfLosses = []     #array to store the differences of the last two losses(for future analysis)

dense1 = Layer_Dense(10, 10)         
activation1 = Activation_ReLU()     

dense2 = Layer_Dense(10, 5)        
activation2 = Activation_ReLU()    

dense3 = Layer_Dense(5, 3)          
activation3 = Activatioin_Softmax() 

denses = [dense1, dense2, dense3]                       #all denses are added to one array, to pass them to future functions
activations = [activation1, activation2, activation3]   

X, y = generateData()

optimiser = Optimiser(X, y, activations, change)       

for i in range(0, iterations):          #cycle which will optimize NN the required number of times
    print(f"Iteration {i}")
    loss = iteration(X, y, denses, activations)
    optimiser.optimise()                #optimising denses
    print(f"Loss {loss}")

    losses.append(loss)                                                                     # appending the loss to the losses array
    if len(losses) > 2 : differenceOfLosses.append(losses[i-1] - losses[i])                 # appends the change in loss to the progressOfLosses array


print(loss)
print(iterations)
makeChart(losses)
makeChart(differenceOfLosses)
input("Press enter to exit")