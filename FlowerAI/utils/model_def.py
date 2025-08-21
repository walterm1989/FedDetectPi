import torch
import torch.nn as nn
import numpy as np
from typing import List

def build_model(num_classes: int = 2, freeze_backbone: bool = True) -> nn.Module:
    import torchvision
    try:
        model = torchvision.models.mobilenet_v3_small(pretrained=True)
    except Exception:
        model = torchvision.models.mobilenet_v3_small(pretrained=False)
    # Replace classifier last linear layer
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    # Freeze backbone if requested
    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith('classifier'):
                param.requires_grad = False
    return model

def get_parameters(model: nn.Module) -> List[np.ndarray]:
    return [p.detach().cpu().numpy() for p in model.state_dict().values()]

def set_parameters(model: nn.Module, params: List[np.ndarray]) -> None:
    state_dict = model.state_dict()
    if len(params) == len(state_dict):
        # strict load by order
        new_state = {}
        for k, v in zip(state_dict.keys(), params):
            new_state[k] = torch.tensor(v)
        model.load_state_dict(new_state, strict=True)
    else:
        # fallback: load by order, strict=False
        new_state = {}
        for k, v in zip(state_dict.keys(), params):
            new_state[k] = torch.tensor(v)
        model.load_state_dict(new_state, strict=False)

def save_ckpt(model: nn.Module, path: str) -> None:
    torch.save(model.state_dict(), path)

def load_ckpt(model: nn.Module, path: str, map_location="cpu") -> bool:
    import os
    if not os.path.exists(path):
        return False
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state, strict=False)
    return True