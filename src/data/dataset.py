"""
PyTorch Dataset for Brain Tumor MRI classification.

Uses ImageFolder format — class is determined by subfolder name.
"""

from pathlib import Path
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from src.data.transforms import get_train_transforms, get_val_transforms


# Maps class names to indices
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]


def get_datasets(data_dir="data/splits"):
    """
    Load train, validation, and test datasets.

    Args:
        data_dir: Path to the splits directory

    Returns:
        Dictionary with 'train', 'val', and 'test' datasets
    """
    data_dir = Path(data_dir)

    datasets = {
        "train": ImageFolder(data_dir / "train", transform=get_train_transforms()),
        "val": ImageFolder(data_dir / "val", transform=get_val_transforms()),
        "test": ImageFolder(data_dir / "test", transform=get_val_transforms()),
    }

    # Verify class order matches our expected order
    for split_name, dataset in datasets.items():
        assert dataset.classes == CLASS_NAMES, (
            f"Class mismatch in {split_name}: "
            f"expected {CLASS_NAMES}, got {dataset.classes}"
        )

    return datasets


def get_dataloaders(data_dir="data/splits", batch_size=16, num_workers=2):
    """
    Create DataLoaders for train, validation, and test sets.

    Args:
        data_dir:    Path to the splits directory
        batch_size:  Number of images per batch
        num_workers: Number of parallel data loading processes

    Returns:
        Dictionary with 'train', 'val', and 'test' DataLoaders
    """
    datasets = get_datasets(data_dir)

    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,        # Randomize order each epoch
            num_workers=num_workers,
            pin_memory=True,     # Faster GPU transfer
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,       # No need to shuffle validation
            num_workers=num_workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }

    return loaders