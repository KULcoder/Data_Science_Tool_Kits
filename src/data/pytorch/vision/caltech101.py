"""
Directly return PyTorch CalTech101 dataloader.
"""

from torchvision import transforms, datasets
from torchvision.transforms import Compose
from torch.utils.data import DataLoader, Dataset, random_split
from torch import Tensor

from typing import Tuple

class GrayscaleToRGB(object):
    def __call__(self, img):
        if img.mode == 'L':
            img = img.convert("RGB")
        return img

def get_transforms() -> Tuple[Compose, Compose]:

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        GrayscaleToRGB(), # because there is some grayscale image mixed in caltech101
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225] # notice, this is stats from ImageNet
        )
    ])
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        GrayscaleToRGB(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
        )
    ])
    # add and modify your own train/test transforms for agumentation
    return train_transforms, test_transforms

def train_val_split(train_set: Dataset, ratio=0.8) -> Tuple[Dataset, Dataset]:
    train_size = int(len(train_set) * ratio)
    val_size = len(train_set) - train_size
    train_subset, val_subset = random_split(train_set, [train_size, val_size])
    return train_subset, val_subset

def get_loaders(bz=128) -> Tuple[DataLoader, DataLoader]:
    root = "./"
    train_transforms, test_transforms = get_transforms()

    dataset = datasets.Caltech101(root=root, download=True, transform=None)
    train_set, test_set = train_val_split(dataset)

    train_set.dataset.transform = train_transforms
    test_set.dataset.transform = test_transforms

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