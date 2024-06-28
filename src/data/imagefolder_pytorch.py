"""
This file serve as a example of how to create a custom dataloader for 
imagefolder in pytorch. 

Directly returns all three kinds of dataloaders!
"""

import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def get_transforms():
    # Custom your transofrms for different datasets!

    train_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])
    validation_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    return train_transforms, validation_transforms, test_transforms

def get_dataset(path):
    train_path = os.path.join(path, 'train')
    test_path = os.path.join(path, 'test')
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Data is not found")

    train_transforms, validation_transforms, test_transforms = get_transforms()

    # train & val, split
    dataset = datasets.ImageFolder(root=train_path, transform=None)
    validation_split = 0.1
    dataset_size = len(dataset)
    val_size = int(validation_split * dataset_size)
    train_size = dataset_size - val_size

    train_dataset, validation_dataset = random_split(dataset, [train_size, val_size])
    train_dataset.dataset.transform = train_transforms
    validation_dataset.dataset.transform = validation_transforms

    # test
    test_dataset = datasets.ImageFolder(root=test_path, transform=test_transforms)

    return train_dataset, validation_dataset, test_dataset


def get_dataloader(path, bz=32):
    # this is the function to be called from
    
    train_dataset, validation_dataset, test_dataset = get_dataset(path)
    train_loader = DataLoader(train_dataset, batch_size=bz, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=bz, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=bz, shuffle=False)

    return train_loader, validation_loader, test_loader