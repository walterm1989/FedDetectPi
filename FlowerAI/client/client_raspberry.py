import os
import torch
import numpy as np
import flwr as fl

from model_def import build_model, set_parameters, get_parameters
from dataset_utils import get_dataloaders

# Set constants
NUM_CLASSES = 2
BATCH_SIZE = 32
LOCAL_EPOCHS = 1
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "latest_global.pt")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data")

# Get server address from environment or use default
SERVER_ADDRESS = os.environ.get("FLOWER_SERVER_ADDR", "127.0.0.1:8080")

# Force CPU
DEVICE = torch.device("cpu")


def load_model():
    model = build_model(num_classes=NUM_CLASSES)
    model.to(DEVICE)
    if os.path.exists(CHECKPOINT_PATH):
        print(f"[INFO] Loading weights from {CHECKPOINT_PATH}")
        state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
    else:
        print("[INFO] No checkpoint found, training from scratch.")
    return model


class RaspberryClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = load_model()

    def get_parameters(self, config=None):
        print("[CLIENT] get_parameters called")
        return get_parameters(self.model)

    def fit(self, parameters, config):
        print("[CLIENT] fit called")
        set_parameters(self.model, parameters)
        train_loader, _, _ = get_dataloaders(
            DATA_PATH, batch_size=BATCH_SIZE, num_workers=0
        )
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        num_examples = 0

        for epoch in range(LOCAL_EPOCHS):
            print(f"[CLIENT] Starting epoch {epoch+1}/{LOCAL_EPOCHS}")
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(DEVICE), target.to(DEVICE)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                num_examples += data.size(0)
                if batch_idx % 10 == 0:
                    print(f"[CLIENT] Batch {batch_idx}, Loss: {loss.item():.4f}")

        # Save latest global parameters after training
        os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
        torch.save(self.model.state_dict(), CHECKPOINT_PATH)
        print(f"[CLIENT] Saved model checkpoint to {CHECKPOINT_PATH}")

        return get_parameters(self.model), num_examples, {}

    def evaluate(self, parameters, config):
        print("[CLIENT] evaluate called")
        set_parameters(self.model, parameters)
        _, val_loader, _ = get_dataloaders(
            DATA_PATH, batch_size=BATCH_SIZE, num_workers=0
        )
        self.model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        loss_total = 0.0
        correct = 0
        num_examples = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = self.model(data)
                loss = criterion(output, target)
                loss_total += loss.item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                num_examples += data.size(0)

        avg_loss = loss_total / num_examples if num_examples > 0 else 0.0
        accuracy = correct / num_examples if num_examples > 0 else 0.0
        print(f"[CLIENT] Evaluation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

        return float(avg_loss), num_examples, {"accuracy": float(accuracy)}


if __name__ == "__main__":
    print(f"[CLIENT] Starting Flower client. Server address: {SERVER_ADDRESS}")
    fl.client.start_numpy_client(server_address=SERVER_ADDRESS, client=RaspberryClient())