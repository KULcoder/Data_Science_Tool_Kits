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
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.5071, 0.4867, 0.4408],
            std = [0.2675, 0.2565, 0.2761]
        )
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.5071, 0.4867, 0.4408],
            std = [0.2675, 0.2565, 0.2761]
        )
    ])
    # add and modify your own train/test transforms for agumentation
    return train_transforms, test_transforms


def get_loaders(bz=256) -> Tuple[DataLoader, DataLoader]:
    root = "./"
    train_transforms, test_transforms = get_transforms()

    train_set = datasets.CIFAR100(root = root, train=True, transform=train_transforms, download=True)
    test_set = datasets.CIFAR100(root = root, train=False, transform=test_transforms, download=True)

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