import os
import torch
from FlowerAI.utils.model_def import save_ckpt, load_ckpt

class ClientRaspberry:
    def __init__(self, model):
        self.model = model
        self.ckpt_path = "model_ckpt.pth"

    def save_model(self):
        save_ckpt(self.model, self.ckpt_path)

    def load_model(self):
        return load_ckpt(self.model, self.ckpt_path)

    def train(self):
        # Dummy training loop for illustration
        for _ in range(2):
            pass
        self.save_model()

    def inference(self, data):
        self.model.eval()
        with torch.no_grad():
            return self.model(data)

if __name__ == "__main__":
    from FlowerAI.utils.model_def import build_model
    model = build_model(num_classes=2, freeze_backbone=True)
    client = ClientRaspberry(model)
    client.train()
    loaded = client.load_model()
    print(f"Model loaded: {loaded}")