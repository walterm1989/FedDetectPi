import inspect
from flwr.server.strategy import FedAvg

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