import numpy as np

def shuffle_data(X, y):
    """
    Shuffle the data (X and y) at the same time.
    For X, shuffle on the first dimension (normally we assume that be the batch/each observation).
    For y, expect one dimension data
    """

    permutation = np.random.permutation(y.shape[0])
    shuffled_X, shuffled_y = X[permutation, :], y[permutation]
    return shuffled_X, shuffled_y

