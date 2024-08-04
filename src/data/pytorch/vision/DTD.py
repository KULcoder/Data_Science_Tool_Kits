"""
Directly return PyTorch DTD dataloader.
"""

from torchvision import transforms, datasets
from torchvision.transforms import Compose
from torch.utils.data import DataLoader, Dataset
from torch import Tensor

from typing import Tuple

def get_transforms() -> Tuple[Compose, Compose]:

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.5305, 0.4752, 0.4269],
            std = [0.2632, 0.2544, 0.2621] 
        )
    ])
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.5286, 0.4703, 0.4254],
            std = [0.2621, 0.2492, 0.2591]
        )
    ])
    # add and modify your own train/test transforms for agumentation
    return train_transforms, test_transforms

def get_loaders(bz=64) -> Tuple[DataLoader, DataLoader]:
    root = "./"
    train_transform, test_transform = get_transforms()

    train_set = datasets.DTD(root=root, split='train', download=True, transform=train_transform)
    test_set = datasets.DTD(root=root, split='test', download=True, transform=test_transform)

    train_loader = DataLoader(
        train_set,
        batch_size = bz,
        shuffle = True
    )

    test_loader = DataLoader(
        test_set,
        batch_size = bz,
        shuffle = False
    )

    return train_loader, test_loader