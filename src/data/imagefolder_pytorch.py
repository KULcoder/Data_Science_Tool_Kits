"""
This file serve as a example of how to create a custom dataloader for 
imagefolder in pytorch. 
"""

import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_transform(split='train'):
    # Custom your transofrms for different datasets!

    if split == 'train':
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
    elif split == 'validation':
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
    elif split == 'test':
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

    return transform

def get_dataset(path, split='train'):
    if split == 'train':
        path = os.path.join(path, 'train')
    elif split == 'validation':
        path = os.path.join(path, 'validation')
    elif split == 'test':
        path = os.path.join(path, 'test')

    transform = get_transform(split)
    return datasets.ImageFolder(root = path, transform = transform)

def get_dataloader(path, split='train', bz=32):
    # this is the function to be called from
    dataset = get_dataset(path, split)
    if split == 'train':
        return DataLoader(dataset, batch_size=bz, shuffle=True)
    elif split == 'validation':
        return DataLoader(dataset, batch_size=bz, shuffle=False)
    elif split == 'test':
        return DataLoader(dataset, batch_size=bz, shuffle=False)