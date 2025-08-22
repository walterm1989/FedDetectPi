import flwr as fl

SERVER_ADDRESS = "0.0.0.0:8080"
ROUNDS = 3

def main():
    strategy = fl.server.strategy.FedAvg()
    print(f"Starting Flower server at {SERVER_ADDRESS} for {ROUNDS} rounds.")
    fl.server.start_server(
        server_address=SERVER_ADDRESS,
        config=fl.server.ServerConfig(num_rounds=ROUNDS),
        strategy=strategy
    )