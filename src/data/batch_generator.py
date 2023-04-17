
def batch_generator(X, y, batch_size):
    """
    Function used to generate minibatches. This method will not shuffle the data.

    inputs:
        X: numpy array of shape (n_samples, n_features)
        y: numpy array of shape (n_samples)

    outputs:
        X_batch: numpy array of shape (batch_size, n_features)
        y_batch: numpy array of shape (batch_size)
    """
    n_samples = X.shape[0]
    for batch_i in range(0, n_samples, batch_size):
        start_i = batch_i
        end_i = start_i + batch_size
        yield X[start_i:end_i], y[start_i:end_i]
