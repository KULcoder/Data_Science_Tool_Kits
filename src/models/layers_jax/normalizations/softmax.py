import jax.numpy as jnp

def softmax(x):
    e_x = jnp.exp(x - jnp.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)