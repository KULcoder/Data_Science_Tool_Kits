"""
Coolects many loss functions and their derivatives.

TODO: Need to be update, and have to think in whih way those functions help calculate the gradients with respect to the weights and biases in a model.
- Cross Entropy Loss
- Hinge Loss
- Mean Squared Error
- Mean Absolute Error
- Binary Cross Entropy
.....
"""

import numpy as np

# Classification related loss classes
class Cross_entropy_loss:
    def __call__(self, y_pred, y):
        """
        Calculate the cross entropy loss.

        inputs:
            y_pred: numpy array of shape (n_samples, n_classes)
            y: numpy array of shape (n_samples)

        outputs:
            loss: float
        """
        n_samples = y.shape[0]
        y_pred = y_pred[np.arange(n_samples), y]
        loss = -np.mean(np.log(y_pred))
        return loss

    def gradient(self, y_pred, y):
        """
        Calculate the gradient of cross entropy loss.

        inputs:
            y_pred: numpy array of shape (n_samples, n_classes)
            y: numpy array of shape (n_samples)

        outputs:
            grad: numpy array of shape (n_samples, n_classes)
        """
        n_samples = y.shape[0]
        grad = y_pred.copy()
        grad[np.arange(n_samples), y] -= 1
        grad = grad / n_samples
        return grad