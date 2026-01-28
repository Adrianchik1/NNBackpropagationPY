import numpy as np

def generateData(inputs=10, outputs=3):
    """
    Generate random training data
    
    :param inputs: Number of input features
    :param outputs: Number of output classes
    :return: X (input data), y (one-hot encoded labels)
    """
    batchSize = 100
    X = np.random.rand(batchSize, inputs)
    y = np.zeros((batchSize, outputs), dtype=int)

    for item in y:
        rand_index = np.random.randint(0, outputs)
        item[rand_index] = 1
    return X, y