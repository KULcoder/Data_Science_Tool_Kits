
import jax.numpy as jnp

def linear_layer(w_b, x):
    W, b = w_b
    return jnp.dot(x, W) + b