import os
import torch
import torchvision.models as models
import numpy as np
from typing import List

def build_model(num_classes: int = 2, freeze_backbone: bool = True) -> torch.nn.Module:
    """
    Build a MobileNetV3 Small model with a custom classifier.

    Args:
        num_classes (int): Number of output classes.
        freeze_backbone (bool): Whether to freeze the backbone parameters.

    Returns:
        torch.nn.Module: The constructed model.
    """
    # Attempt to load pretrained model, fallback to untrained if fails
    try:
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    except Exception:
        model = models.mobilenet_v3_small(weights=None)
    # Replace classifier's last layer
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_features, num_classes)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True

    return model

def get_parameters(model) -> List[np.ndarray]:
    """
    Get the model parameters as a list of numpy arrays.

    Args:
        model (torch.nn.Module): The model.

    Returns:
        List[np.ndarray]: List of parameter arrays.
    """
    state_dict = model.state_dict()
    return [v.cpu().numpy() for v in state_dict.values()]

def set_parameters(model, params: List[np.ndarray]) -> None:
    """
    Set the model parameters from a list of numpy arrays.

    Args:
        model (torch.nn.Module): The model.
        params (List[np.ndarray]): List of parameter arrays.

    Returns:
        None
    """
    state_dict = model.state_dict()
    keys = list(state_dict.keys())
    if len(params) == len(keys):
        # Direct mapping by order, strict=True
        new_state = {k: torch.from_numpy(v).to(state_dict[k].device) for k, v in zip(keys, params)}
        model.load_state_dict(new_state, strict=True)
    else:
        # Partial mapping by order, strict=False
        new_state = {}
        for i, (k, v) in enumerate(zip(keys, params)):
            new_state[k] = torch.from_numpy(v).to(state_dict[k].device)
        state_dict.update(new_state)
        model.load_state_dict(state_dict, strict=False)

def save_ckpt(model, path: str) -> bool:
    """
    Save the model's state_dict to the given path.

    Args:
        model (torch.nn.Module): The model.
        path (str): File path to save the checkpoint.

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(model.state_dict(), path)
        return True
    except Exception:
        return False

def load_ckpt(model, path: str, map_location: str = 'cpu') -> bool:
    """
    Load the model's state_dict from the given path.

    Args:
        model (torch.nn.Module): The model.
        path (str): File path of the checkpoint.
        map_location (str): Device to map the checkpoint.

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        state = torch.load(path, map_location=map_location)
        model.load_state_dict(state)
        return True
    except Exception:
        return False