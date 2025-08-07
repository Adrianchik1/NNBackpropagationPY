class CalculateA:
    a = []
    def __init__(self, X):
        X = X.T
        CalculateA.a = [X]
    def calculateA(self, X):
        X = X.T
        CalculateA.a.append(X)