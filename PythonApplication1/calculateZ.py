class CalculateZ:
    CalculateZInstances = []
    def __init__(self):
        self.z = []
        self.CalculateZInstances.append(self)
    def calculateZ(self, X):
        X = X.T
        self.z.append(X)