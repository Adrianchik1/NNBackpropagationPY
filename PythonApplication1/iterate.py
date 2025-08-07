from losses import MSELoss
from calculateZ import CalculateZ
from losses import MSELoss
from calculateA import CalculateA

def iteration(X, y, denses, activations):
    weightOfNeurons = CalculateZ()
    weightOFNeuronsActivated = CalculateA(X)
    for dense, activation in zip(denses, activations):
        dense.forward(X)

        weightOfNeurons.calculateZ(dense.output)

        activation.forward(dense.output)
        X = activation.output
        weightOFNeuronsActivated.calculateA(X)
    lossFunction = MSELoss()
    loss = lossFunction.calculate(X, y)

    return loss
