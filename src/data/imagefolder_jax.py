"""
Create data pipeline for jax usage using local files in imagefolder structures:

train
    -class 1
        -image 1
        -image 2
        ...
    -class 2
    ...
test
    -class 1
    ...
"""

import jax
from PIL import Image
import numpy as np
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds

# Enable JAX to use GPU

tf.config.experimental.set_visible_devices([], "GPU")

# Define preprocessing function
# can be customed for different datasets and validation/test sets
def preprocess(image, label):
     image = tf.image.resize(image, [128, 128]) # resize
     image = tf.cast(image, tf.float32) / 255.0 # normalize
     return image, label 

def load_image(path):
    image = Image.open(path)
    return np.array(image)

# loader
def load_dataset(batch_size):
     dataset = tfds.load('your_dataset_name', split='train', as_supervised=True)
     dataset = dataset.map(preprocess).batch(batch_size).prefetch(tf.data.AUTOTUNE)
     return dataset

# convert TensorFlow dataset to JAX numpy arrays
def tf_to_jax(data):
     for image, label in data:
          yield jax.device_put((jnp.array(image), jnp.array(label)))

# Example of using this data loader
batch_size = 32
dataset = load_dataset(batch_size)
jax_dataset = tf_to_jax(dataset)

for images, labels in jax_dataset:
     print(images.shape, labels.shape)

