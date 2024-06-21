import numpy as np

def accuracy(y_pred, y):
    """
    Calculate the accuracy of the model.
    """
    return np.mean(y_pred == y)