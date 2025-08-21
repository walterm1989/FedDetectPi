import inspect
import flwr as fl
from flwr.server.strategy import FedAvg
import torch
from FlowerAI.utils.model_def import build_model, save_ckpt

class MyStrategy(FedAvg):
    def __init__(self, **kwargs):
        # Handle possible parameter name difference: min_evaluate_clients vs min_eval_clients
        sig = inspect.signature(FedAvg.__init__)
        params = sig.parameters
        if "min_evaluate_clients" in params:
            if "min_eval_clients" in kwargs:
                kwargs["min_evaluate_clients"] = kwargs.pop("min_eval_clients")
        elif "min_eval_clients" in params:
            if "min_evaluate_clients" in kwargs:
                kwargs["min_eval_clients"] = kwargs.pop("min_evaluate_clients")
        super().__init__(**kwargs)

def save_global_model(weights, path="FlowerAI/checkpoints/server_model.pth"):
    model = build_model(num_classes=2, freeze_backbone=False)
    state_dict = model.state_dict()
    for k, v in zip(state_dict.keys(), weights):
        state_dict[k] = torch.tensor(v)
    model.load_state_dict(state_dict)
    save_ckpt(model, path)

def main():
    strategy = MyStrategy()
    history = fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=3),
    )
    # After training, save the global model
    if hasattr(strategy, "parameters") and strategy.parameters is not None:
        save_global_model(strategy.parameters)

if __name__ == "__main__":
    main()