"""
FlowerAI Server main module.
"""

import os
from typing import Optional, List, Tuple, Dict, Any
import flwr as fl
import numpy as np

from FlowerAI.utils.model_def import build_model, set_parameters, save_ckpt
import torch

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

class MyStrategy(fl.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        # Handle both min_evaluate_clients (old) and min_eval_clients (new)
        if "min_eval_clients" in kwargs:
            kwargs["min_evaluate_clients"] = kwargs.pop("min_eval_clients")
        super().__init__(**kwargs)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Optional[fl.common.Parameters]:
        aggregated_parameters = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is not None:
            # Convert parameters to ndarrays
            ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)
            # Rebuild model and load parameters
            model = build_model(num_classes=2, freeze_backbone=False)
            set_parameters(model, ndarrays)
            # Save checkpoint
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"global_round_{server_round}.pt")
            save_ckpt(model, ckpt_path)
        return aggregated_parameters

def main():
    """
    Main function for starting the FlowerAI server.
    """
    strategy = MyStrategy(
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        on_fit_config_fn=None,
        on_evaluate_config_fn=None,
    )
    print("Starting Flower server on 0.0.0.0:8080 with 3 rounds...")
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()