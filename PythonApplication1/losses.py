import numpy as np

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossentrophy(Loss): 
    def forward(self, predictions, targets):
        samples = len(predictions)
        predictions_clipped = np.clip(predictions, 1e-7, 1-1e-7)

        if len(targets.shape) == 1:
            correct_confidences = predictions_clipped[range(samples), targets]
        elif len(targets.shape) == 2:
            correct_confidences = np.sum(predictions_clipped*targets, axis = 1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

class MSELoss(Loss):
    diff = None
    # MSELossInstances = []
    # def __init__(self):
    #     self.MSELossInstances.append(self)
    def forward(self, predictions, targets):
        predictions = np.array(predictions)
        targets = np.array(targets)

        MSELoss.diff = predictions - targets
        mse = np.mean(self.diff ** 2)
        return mse
    @staticmethod
    def backward():
        # diff_single = MSELoss.diff[0]  # shape (output_size,)
        # delta = (2 * diff_single / len(MSELoss.diff)).reshape(-1, 1)  # shape (output_size, 1)
        #return delta
        return (2 * MSELoss.diff / len(MSELoss.diff)).T
