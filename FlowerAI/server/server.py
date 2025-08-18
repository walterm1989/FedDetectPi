import os
import flwr as fl
import torch
import logging

# --- Server configuration ---
SERVER_ADDRESS = "[::]:8080"
ROUNDS = 10
MIN_FIT = 2
MIN_EVAL = 2
MIN_AVAILABLE = 2
CHECKPOINT_DIR = "checkpoints"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

def ensure_checkpoint_dir():
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

class MyStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        # Call FedAvg's aggregate_fit to get the aggregated weights
        aggregated_fit = super().aggregate_fit(server_round, results, failures)
        if aggregated_fit is not None:
            parameters, metrics = aggregated_fit
            # Convert Flower parameters to PyTorch state_dict format
            weights = fl.common.parameters_to_ndarrays(parameters)
            # Save as PyTorch tensor list
            ensure_checkpoint_dir()
            save_path = os.path.join(CHECKPOINT_DIR, f"global_weights_round_{server_round}.pt")
            torch.save(weights, save_path)
            logging.info(f"Saved global weights to {save_path} at end of round {server_round}.")
        else:
            logging.warning("No aggregated weights to save at end of round %d.", server_round)
        return aggregated_fit

def main():
    logging.info("Starting Flower server at %s for %d rounds.", SERVER_ADDRESS, ROUNDS)
    strategy = MyStrategy(
        min_fit_clients=MIN_FIT,
        min_eval_clients=MIN_EVAL,
        min_available_clients=MIN_AVAILABLE,
    )
    fl.server.start_server(
        server_address=SERVER_ADDRESS,
        config=fl.server.ServerConfig(num_rounds=ROUNDS),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()