"""
Coolects many loss functions.
Since JAX has autograd, we will not implement derivative.

"""

import numpy as np
import jax.numpy as jnp
import jax

# # Old numpy implementation
# # Classification related loss classes
# class Cross_entropy_loss:
#     def __call__(self, y_pred, y):
#         """
#         Calculate the cross entropy loss.

#         inputs:
#             y_pred: numpy array of shape (n_samples, n_classes)
#             y: numpy array of shape (n_samples)

#         outputs:
#             loss: float
#         """
#         n_samples = y.shape[0]
#         y_pred = y_pred[np.arange(n_samples), y]
#         loss = -np.mean(np.log(y_pred))
#         return loss

#     def gradient(self, y_pred, y):
#         """
#         Calculate the gradient of cross entropy loss.

#         inputs:
#             y_pred: numpy array of shape (n_samples, n_classes)
#             y: numpy array of shape (n_samples)

#         outputs:
#             grad: numpy array of shape (n_samples, n_classes)
#         """
#         n_samples = y.shape[0]
#         grad = y_pred.copy()
#         grad[np.arange(n_samples), y] -= 1
#         grad = grad / n_samples
#         return grad

def binary_cross_entropy_loss(predictions, labels):
    """
    Compute the binary cross entropy loss.

    args:
        predictions: jnp array of shape (n_samples, ), this shouls already be 
            the probability rather than logits
        labels: jnp array of shape (n_samples, )

    returns:
        cross_entropy_loss, loss
    """

    epsilon = 1e-15
    predictions = jnp.clip(predictions, epsilon, 1 - epsilon)

    loss = -jnp.mean(labels * jnp.log(predictions) + (1 - labels) * jnp.log(1 - predictions))
    return loss
    

def cross_entropy_loss(logits, labels):
    """
    Calculate the cross entropy loss.

    args:
        logits: jnp array of shape (n_samples, n_classes)
        labels: jnp array of shape (n_samples, )

    returns:
        cross_entropy_loss, loss
    """
    num_classes = logits.shape[1]
    one_hot_labels = jax.nn.one_hot(labels, num_classes)
    probs = jax.nn.softmax(logits)

    # compute the cross-entropy loss for each instance in the batch
    batch_losses = -jnp.sum(one_hot_labels * jnp.log(probs), axis=1)
    mean_cross_entropy_loss = jnp.mean(batch_losses)
    return mean_cross_entropy_loss

def hinge_loss(logits, labels):
    """
    Computes the hinge loss.

    args:
        predictions: jnp array of shape (n_samples, num_classes)
        targets: jnp array of shape (n_samples, )
    
    returns:
        hinge_loss
    
    """

    num_classes = logits.shape[1]
    one_hot_targets = jax.nn.one_hot(labels, num_classes) * 2 - 1

    hinge_losses = jnp.maximum(0, 1 - one_hot_targets * logits)
    mean_hinge_loss = jnp.mean(jnp.sum(hinge_losses, axis=1))

    return mean_hinge_loss

def mean_squared_loss(predictions, targets):
    """
    Calculate the mean squared loss.

    args:
        predictions: jnp array of shape (n_samples, num_outputs)
        targets: jnp array of shape (n_samples, num_outputs)
    
    returns:
        mse_loss
    """

    squared_diff = jnp.square(predictions - targets)
    mse_loss = jnp.mean(squared_diff)

    return mse_loss

def mean_absolute_loss(predictions, targets):
    """
    Calculate the mean absolute loss.

    args:
        predictions: jnp array of shape (n_samples, num_outputs)
        targets: jnp array of shape (n_samples, num_outputs)
    
    returns:
        mae_loss
    """

    absolute_diff = jnp.abs(predictions - targets)
    mae_loss = jnp.mean(absolute_diff)

    return mae_loss



