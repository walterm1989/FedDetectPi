"""
Model definition and checkpoint utilities for FlowerAI.
"""

from typing import List
import torch
import numpy as np
from torchvision import models
import os

def build_model(num_classes: int = 2, freeze_backbone: bool = True):
    """
    Build a MobileNetV3 Small model for image classification.
    Args:
        num_classes (int): Number of output classes.
        freeze_backbone (bool): If True, freeze all backbone parameters except classifier.
    Returns:
        torch.nn.Module: Configured MobileNetV3 Small model.
    """
    # Try to use pretrained weights if available
    try:
        # torchvision >= 0.13
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    except Exception:
        # Fallback for earlier versions
        model = models.mobilenet_v3_small(pretrained=True)

    # Replace classifier
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_features, num_classes)

    # Optionally freeze backbone
    if freeze_backbone:
        for name, param in model.named_parameters():
            # Only classifier params remain trainable
            if not name.startswith("classifier"):
                param.requires_grad = False

    return model

def get_parameters(model) -> List[np.ndarray]:
    """
    Get model parameters as a list of numpy arrays.
    Args:
        model (torch.nn.Module)
    Returns:
        List[np.ndarray]: List of parameter arrays.
    """
    return [param.detach().cpu().numpy() for param in model.state_dict().values()]

def set_parameters(model, params: List[np.ndarray]):
    """
    Set model parameters from a list of numpy arrays (ordered as state_dict).
    Args:
        model (torch.nn.Module)
        params (List[np.ndarray])
    """
    state_dict = model.state_dict()
    if len(params) != len(state_dict):
        raise ValueError("Number of parameters does not match model state_dict.")
    new_state_dict = {}
    for (k, v), np_val in zip(state_dict.items(), params):
        tensor = torch.from_numpy(np_val)
        # Ensure type and shape match
        if v.shape != tensor.shape:
            raise ValueError(f"Shape mismatch for {k}: {v.shape} vs {tensor.shape}")
        new_state_dict[k] = tensor.type_as(v)
    model.load_state_dict(new_state_dict, strict=True)

def save_ckpt(model, path: str):
    """
    Save model checkpoint to the given path.
    Args:
        model (torch.nn.Module)
        path (str): File path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def load_ckpt(model, path: str, map_location: str = 'cpu') -> bool:
    """
    Load model checkpoint from the given path.
    Args:
        model (torch.nn.Module)
        path (str): File path.
        map_location: torch device string for loading
    Returns:
        bool: True if loaded successfully, False if file missing
    """
    if not os.path.isfile(path):
        return False
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state)
    return True