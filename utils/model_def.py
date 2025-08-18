import os
import torch
import numpy as np
import torchvision.models as models

def build_model(num_classes=2, freeze_backbone=True):
    """
    Build a MobileNetV3 Small model with customizable classifier and backbone freezing.

    Args:
        num_classes (int): Number of output classes for classification.
        freeze_backbone (bool): If True, freeze all backbone parameters.

    Returns:
        torch.nn.Module: The modified MobileNetV3 Small model.
    """
    # Load pretrained MobileNetV3 Small
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    # Modify the classifier for the required number of classes
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
    # Freeze backbone if requested (all except classifier)
    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("classifier"):
                param.requires_grad = False
    return model

def get_parameters(model):
    """
    Extract model parameters as a list of NumPy arrays.

    Args:
        model (torch.nn.Module): The model to extract parameters from.

    Returns:
        List[np.ndarray]: List of parameter arrays (CPU, contiguous).
    """
    return [p.detach().cpu().numpy().copy() for p in model.parameters()]

def set_parameters(model, params):
    """
    Set model parameters from a list of NumPy arrays.

    Args:
        model (torch.nn.Module): The model to set parameters for.
        params (List[np.ndarray]): List of parameter arrays.
    """
    with torch.no_grad():
        for p, arr in zip(model.parameters(), params):
            p.copy_(torch.from_numpy(arr).to(p.device))

def save_ckpt(model, path):
    """
    Save the model state dict to a file.

    Args:
        model (torch.nn.Module): The model to save.
        path (str): Path to save the checkpoint.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def load_ckpt(model, path):
    """
    Load model state dict from a file.

    Args:
        model (torch.nn.Module): The model to load parameters into.
        path (str): Path to the checkpoint file.
    """
    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict)