import numpy as np

def z_score_normalize(X, epsilon=1e-8):
    # X here should be already the numpy array
    std = X.std(axis=1, keepdims=True)
    mean = X.mean(axis=1, keepdims=True)
    X = (X - mean) / (std + epsilon)
    return X

def min_max_normalize(X, epsilon=1e-8):
    # X here should be already the numpy array
    X_min = X.min(axis=1, keepdims=True)
    X_max = X.max(axis=1, keepdims=True)

    X = (X - X_min) / (X_max - X_min + epsilon)
    return X