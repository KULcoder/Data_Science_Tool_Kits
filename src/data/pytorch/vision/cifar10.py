"""
Directly return PyTorch CIFAR10 dataloader.
"""

from torchvision import transforms, datasets
from torchvision.transforms import Compose
from torch.utils.data import DataLoader, Dataset
from torch import Tensor

from typing import Tuple

def get_transforms() -> Tuple[Compose, Compose]:
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4915, 0.4823, .4468), 
            (0.2470, 0.2435, 0.261)
        )
    ])
    test_transforms = train_transforms
    # add and modify your own train/test transforms for agumentation
    return train_transforms, test_transforms


def get_loaders(bz=256) -> Tuple[DataLoader, DataLoader]:
    root = "./"
    train_transforms, test_transforms = get_transforms()

    train_set = datasets.CIFAR10(root = root, train=True, transform=train_transforms, download=True)
    test_set = datasets.CIFAR10(root = root, train=False, transform=test_transforms, download=True)

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