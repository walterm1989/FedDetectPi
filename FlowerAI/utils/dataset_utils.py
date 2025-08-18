import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Define the transform pipeline as specified
transform = transforms.Compose([
    transforms.Resize(224),          # Resize shorter edge to 224
    transforms.CenterCrop(224),      # Center crop to 224x224
    transforms.ToTensor(),           # Convert PIL image to tensor
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

def get_dataloaders(data_root, batch_size=32, val_split=0.2, num_workers=0):
    """
    Loads an ImageFolder dataset from `data_root`, splits it into training and validation sets,
    and returns corresponding DataLoaders.

    Args:
        data_root (str): Path to the root directory containing image folders.
        batch_size (int): Batch size for the DataLoaders.
        val_split (float): Fraction of the dataset to use as validation.
        num_workers (int): Number of subprocesses to use for data loading.

    Returns:
        train_loader (DataLoader): DataLoader for the training split.
        val_loader (DataLoader): DataLoader for the validation split.
    """
    # Ensure reproducibility
    generator = torch.Generator().manual_seed(42)

    dataset = datasets.ImageFolder(root=data_root, transform=transform)
    n_total = len(dataset)
    n_val = int(val_split * n_total)
    n_train = n_total - n_val

    train_dataset, val_dataset = random_split(dataset, [n_train, n_val], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader