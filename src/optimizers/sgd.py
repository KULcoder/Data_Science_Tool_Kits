"""
designed for JAX sgd.
"""

def sgd(params, grads, learning_rate):
    if type(params) == list:
        return [(W - learning_rate * dW, b - learning_rate * db) for (W, b), (dW, db) in zip(params, grads)]
    else:
        return params - learning_rate * grads