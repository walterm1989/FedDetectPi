import flwr as fl
import numpy as np
import torch
from FlowerAI.utils.model_def import build_model, save_ckpt, load_ckpt

class FlowerClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = build_model(num_classes=2, freeze_backbone=True)
        self.ckpt_path = "model_ckpt.pth"
        load_ckpt(self.model, self.ckpt_path)

    def get_parameters(self, config=None):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        for k, v in zip(state_dict.keys(), parameters):
            state_dict[k] = torch.tensor(v)
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config=None):
        self.set_parameters(parameters)
        # Dummy fit, real training logic goes here
        save_ckpt(self.model, self.ckpt_path)
        return self.get_parameters(), len(parameters), {}

    def evaluate(self, parameters, config=None):
        self.set_parameters(parameters)
        # Dummy evaluation, real evaluation logic goes here
        loss, accuracy = 0.0, 0.0
        return float(loss), len(parameters), {"accuracy": float(accuracy)}

if __name__ == "__main__":
    fl.client.start_numpy_client(server_address="localhost:8080", client=FlowerClient())