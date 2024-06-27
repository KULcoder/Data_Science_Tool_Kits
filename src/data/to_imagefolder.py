"""
NOT TESTED

The following code provides script to arrange MNIST and CIFAR10/100 dataset
into standard imagefolder file structure. This script using tensorflow datasets
and saving images into png files. Not very efficient.

How to use:

python3 to_imagefolder.py <dataset_name>
"""

import os
import sys
import numpy as np
from tensorflow.keras.datasets import mnist, cifar10, cifar100
from PIL import Image


def save_images(images, labels, folder_name, is_cifar=False):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        
    for i, (image, label) in enumerate(zip(images, labels)):
        if is_cifar:
            label = label[0] # Extract label from array
        label_folder = os.path.join(folder_name, str(label))
        
        if not os.path.exists(label_folder):
            os.makedirs(label_folder)
            
        file_path = os.path.join(label_folder, f"{i}.png")
        Image.fromarray(image).save(file_path)

if __name__ == '__main__':

    # try to understand what dataset wish to transform
    dataset_name = sys.argv[1]

    is_cifar = True if (dataset_name == 'cifar10' or dataset_name == 'cifar100') else False
    folder_name = dataset_name

    if dataset_name == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    elif dataset_name == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    elif dataset_name == 'cifar100':
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    else:
        raise NotImplementedError("Dataset is not implemented")
    
    folder_name_train = os.path.join(folder_name, "train")
    folder_name_test = os.path.join(folder_name, 'test')
    save_images(x_train, y_train, folder_name_train, is_cifar)
    save_images(x_test, y_test, folder_name_test, is_cifar)

