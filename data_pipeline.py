import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_transforms(enable_augmentation: bool = True):
    """
    Build preprocessing and augmentation pipelines for MNIST train/test sets.
    """
    normalize = transforms.Normalize((0.1307,), (0.3081,))

    if enable_augmentation:
        train_transform = transforms.Compose(
            [
                transforms.RandomRotation(degrees=12),
                transforms.RandomAffine(
                    degrees=0, translate=(0.08, 0.08), scale=(0.95, 1.05)
                ),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )

    return train_transform, test_transform


def build_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 0,
    enable_augmentation: bool = True,
):
    """
    Load MNIST and return train/test dataloaders.
    """
    train_transform, test_transform = build_transforms(
        enable_augmentation=enable_augmentation
    )

    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform,
    )

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader
