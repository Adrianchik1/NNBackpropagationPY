from losses import Loss_CategoricalCrossentrophy

def iteration(X, y, denses, activations):
    for dense, activation in zip(denses, activations):
        dense.forward(X)
        activation.forward(dense.output)
        X = activation.output

    loss_function = Loss_CategoricalCrossentrophy()
    loss = loss_function.calculate(X, y)

    return loss

#LOX