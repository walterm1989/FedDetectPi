"""
NumPyClient for Raspberry Pi (RPi) devices.
"""

import os
import flwr as fl
import numpy as np
import torch
from FlowerAI.utils.model_def import build_model, get_parameters, set_parameters, save_ckpt, load_ckpt
from FlowerAI.utils.dataset_utils import get_dataloaders

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints")
CKPT_PATH = os.path.join(CHECKPOINT_DIR, "latest_global.pt")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

class RaspberryClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = build_model(num_classes=2, freeze_backbone=False)
        if os.path.exists(CKPT_PATH):
            load_ckpt(self.model, CKPT_PATH, strict=False)

    def get_parameters(self, config=None):
        return get_parameters(self.model)

    def fit(self, parameters, config=None):
        set_parameters(self.model, parameters)
        batch_size = int(config["batch_size"]) if config and "batch_size" in config else 32
        epochs = int(config["local_epochs"]) if config and "local_epochs" in config else 2

        train_loader, val_loader = get_dataloaders(
            data_root=os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"),
            batch_size=batch_size,
            val_split=0.2,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()
        for epoch in range(epochs):
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                out = self.model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
        save_ckpt(self.model, CKPT_PATH)
        return get_parameters(self.model), len(train_loader.dataset), {}

    def evaluate(self, parameters, config=None):
        set_parameters(self.model, parameters)
        _, val_loader = get_dataloaders(
            data_root=os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"),
            batch_size=32,
            val_split=0.2,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = self.model(xb)
                loss = criterion(out, yb)
                total_loss += loss.item() * xb.size(0)
                pred = torch.argmax(out, dim=1)
                correct += (pred == yb).sum().item()
                total += xb.size(0)
        loss = total_loss / total if total > 0 else float("nan")
        acc = correct / total if total > 0 else 0.0
        return float(loss), len(val_loader.dataset), {"accuracy": acc}

def main():
    server_addr = os.environ.get("FLOWER_SERVER_ADDR", "0.0.0.0:8080")
    client = RaspberryClient()
    fl.client.start_numpy_client(server_address=server_addr, client=client)

if __name__ == "__main__":
    main()