"""
Dataset utilities for FlowerAI.
"""

from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
import torch

def get_dataloaders(
    data_root: str,
    batch_size: int = 32,
    val_split: float = 0.2,
    num_workers: int = 0
):
    """
    Create train and validation dataloaders from an ImageFolder directory.
    Args:
        data_root (str): Path to data directory with subfolders per class.
        batch_size (int): Batch size for loaders.
        val_split (float): Fraction of data for validation (rest for train).
        num_workers (int): DataLoader workers.
    Returns:
        (train_loader, val_loader, class_names)
    """

    # Standard ImageNet normalization
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    transform = transforms.Compose([
        transforms.Resize(224),  # Resize shorter side to 224
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    dataset = datasets.ImageFolder(data_root, transform=transform)
    num_total = len(dataset)
    num_val = int(val_split * num_total)
    num_train = num_total - num_val

    # Fixed split for reproducibility
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [num_train, num_val], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, dataset.classes