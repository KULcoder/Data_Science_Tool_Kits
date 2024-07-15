"""
NOT TESTED.
Using JAX, implement the MLP with pytorch style.
"""
import sys, os
import jax
import jax.numpy as jnp
from jax import grad, random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from models.layers.activations import relu
from models.layers.normalizations.softmax import softmax
from models.layers.linears import linear_layer
from criterion.accuracy import compute_accuracy
from optimizers.sgd import sgd
from data.mnist_tensorflow import load_data

def cross_entropy_loss(params, x, y):
    logits = mlp(params, x)
    return -jnp.mean(jnp.sum(y * jnp.log(softmax(logits)), axis=1))

def create_params(layer_sizes, key=random.PRNGKey(42)):
    # init random weights
    params = []

    keys = random.split(key, len(layer_sizes) - 1)
    for m, n, k in zip(layer_sizes[:-1], layer_sizes[1:], keys):
        # [:-1] excluding the last, [1:] excluding the first
        W = random.normal(k, (m, n)) * jnp.sqrt(2.0 / m)
        b = jnp.zeros(n)
        params.append((W, b))

    return params

def mlp(params, x):
    for W, b in params[:-1]:
        x = relu(linear_layer((W, b), x))
    W, b = params[-1]
    return linear_layer((W, b), x)    

# training loop
def train_mlp(params, epochs=10, learning_rate=1e-2):
    train_ds, test_ds = load_data()

    for epoch in range(epochs):
        train_loss = 0
        number_batches = 0
        for batch in train_ds:
            images, labels = batch['image'], batch['label']
            logits = mlp(params, images)

            loss, loss_grad = jax.value_and_grad(cross_entropy_loss)(params, logits, labels)
            params = sgd(params, loss_grad, learning_rate)

            train_loss += loss
            number_batches += 1

        print('Train loss:', train_loss/number_batches)

    test_accs = 0.0
    num_batches = 0
    for batch in test_ds:
        images, labels = batch['image'], batch['label']
        logits = mlp(params, images)

        test_acc = compute_accuracy(logits, labels)

        test_accs += test_acc
        num_batches += 1
    print("Test ACC:", test_accs / num_batches)
    
if __name__ == '__main__':
    layer_sizes = [3072, 128, 10]
    params = create_params(layer_sizes)
    train_mlp(params)
        
    


