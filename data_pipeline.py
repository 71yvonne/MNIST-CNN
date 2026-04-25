import http.client
import time
import urllib.error

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

    # Some default upstream URLs are occasionally flaky in certain regions.
    # Add extra mirrors and retry downloads for transient network failures.
    datasets.MNIST.mirrors = [
        "https://ossci-datasets.s3.amazonaws.com/mnist/",
        "https://storage.googleapis.com/cvdf-datasets/mnist/",
        "http://yann.lecun.com/exdb/mnist/",
    ]

    def _build_mnist_dataset(train: bool, transform):
        max_retries = 3
        retryable_errors = (
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            http.client.RemoteDisconnected,
            OSError,
        )
        for attempt in range(1, max_retries + 1):
            try:
                return datasets.MNIST(
                    root=data_dir,
                    train=train,
                    download=True,
                    transform=transform,
                )
            except retryable_errors as exc:
                if attempt == max_retries:
                    split = "train" if train else "test"
                    raise RuntimeError(
                        f"Failed to download MNIST {split} set after {max_retries} attempts. "
                        "Please check network/proxy, or pre-download MNIST into data/MNIST/raw."
                    ) from exc
                time.sleep(attempt)

    train_dataset = _build_mnist_dataset(train=True, transform=train_transform)
    test_dataset = _build_mnist_dataset(train=False, transform=test_transform)

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
