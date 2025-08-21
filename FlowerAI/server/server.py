import inspect
import os
import flwr as fl
from flwr.server.strategy import FedAvg
import torch
from FlowerAI.utils import build_model, save_ckpt
import flwr.common

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

    def aggregate_fit(self, rnd, results, failures):
        agg_result = super().aggregate_fit(rnd, results, failures)
        if agg_result is None:
            return None
        parameters, metrics = agg_result

        # Convert parameters to ndarrays
        ndarrays = flwr.common.parameters_to_ndarrays(parameters)

        # Build the model
        model = build_model(num_classes=2, freeze_backbone=False)
        state_dict = model.state_dict()
        # Map ndarrays to state_dict keys, allow strict=False
        for k, w in zip(state_dict.keys(), ndarrays):
            state_dict[k] = torch.tensor(w)
        model.load_state_dict(state_dict, strict=False)

        # Ensure checkpoints directory exists
        ckpt_dir = os.path.join("FlowerAI", "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f"global_round_{rnd}.pt")
        save_ckpt(model, ckpt_path)

        return parameters, metrics

def main():
    strategy = MyStrategy()
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=3),
    )

if __name__ == "__main__":
    main()