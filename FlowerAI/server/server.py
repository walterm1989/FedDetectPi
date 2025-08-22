from __future__ import annotations
import os
import flwr as fl

SERVER_ADDRESS = os.getenv("FLOWER_SERVER_ADDRESS", "0.0.0.0:8080")
ROUNDS = int(os.getenv("FLOWER_NUM_ROUNDS", "3"))

def main() -> None:
    print(f"[FlowerAI] Starting Flower server at {SERVER_ADDRESS} for {ROUNDS} rounds.")
    strategy = fl.server.strategy.FedAvg()
    fl.server.start_server(
        server_address=SERVER_ADDRESS,
        config=fl.server.ServerConfig(num_rounds=ROUNDS),
        strategy=strategy,
    )
    print("[FlowerAI] Server stopped.")

if __name__ == "__main__":
    main()