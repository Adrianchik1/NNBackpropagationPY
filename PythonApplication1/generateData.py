import numpy as np

def generateData():
    batchSize, inputs = 100, 16
    outputs = 5
    X = np.random.rand(batchSize, inputs)
    y = np.zeros((batchSize, outputs), dtype=int)

    for item in y:
        rand_index = np.random.randint(0, outputs)
        item[rand_index] = 1
    return X, y