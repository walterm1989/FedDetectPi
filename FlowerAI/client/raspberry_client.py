from __future__ import annotations
import os
import flwr as fl
import numpy as np

# Dirección del servidor (editar por variable de entorno en la RPi)
SERVER_ADDRESS_DEFAULT = "127.0.0.1:8080"  # se sobrescribe con FLOWER_SERVER_ADDRESS

class PassiveClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        # Devolver un vector dummy para inicializar parámetros
        return [np.zeros(1, dtype=np.float32)]

    def fit(self, parameters, config):
        print(f"[FlowerAI][RPi] Fit -> params: {len(parameters)} | sin entrenamiento local")
        # No entrenamos nada; devolvemos lo recibido y 0 ejemplos
        return parameters, 0, {}

    def evaluate(self, parameters, config):
        print("[FlowerAI][RPi] Evaluate -> sin datos locales")
        # Sin datos: loss=0.0 y 0 ejemplos
        return 0.0, 0, {}


def main() -> None:
    addr = os.getenv("FLOWER_SERVER_ADDRESS", SERVER_ADDRESS_DEFAULT)
    print(f"[FlowerAI][RPi] Conectando con servidor en {addr}")
    fl.client.start_numpy_client(server_address=addr, client=PassiveClient())

if __name__ == "__main__":
    main()