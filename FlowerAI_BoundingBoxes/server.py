#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Servidor Flower (plano de control):
# Publica parámetros de configuración a los clientes (no hay entrenamiento).
# Parámetros publicados:
#   - conf_thr (float)
#   - nms_thr (float)
#   - input_size (int)
#   - person_class_name (str) = "person"
#
# Ejemplo:
#   python FlowerAI_BoundingBoxes/server.py \
#     --address 0.0.0.0:8080 --rounds 9999 \
#     --conf-thr 0.40 --nms-thr 0.45 --input-size 416

import argparse
from typing import Dict

import flwr as fl


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Servidor Flower - Plano de control BBoxes YOLOv4-tiny")
    p.add_argument("--address", type=str, default="0.0.0.0:8080", help="Dirección host:puerto del servidor Flower")
    p.add_argument("--rounds", type=int, default=9999, help="Número de rondas (usar grande para modo continuo)")
    p.add_argument("--conf-thr", type=float, default=0.40, help="Umbral de confianza")
    p.add_argument("--nms-thr", type=float, default=0.45, help="Umbral de NMS")
    p.add_argument("--input-size", type=int, default=416, choices=[320, 416], help="Tamaño de entrada cuadrado")
    return p


def make_on_fit_config_fn(conf_thr: float, nms_thr: float, input_size: int):
    # Devuelve una función que Flower invocará al inicio de cada ronda
    def on_fit_config_fn(server_round: int) -> Dict[str, float]:
        # Publicamos configuración a los clientes
        return {
            "conf_thr": float(conf_thr),
            "nms_thr": float(nms_thr),
            "input_size": int(input_size),
            "person_class_name": "person",
            "server_round": int(server_round),
        }

    return on_fit_config_fn


def main() -> None:
    args = build_argparser().parse_args()

    on_fit_config_fn = make_on_fit_config_fn(args.conf_thr, args.nms_thr, args.input_size)

    # Estrategia FedAvg básica, solo para canal de control
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.0,          # no requerimos entrenamiento
        fraction_evaluate=0.0,     # no evaluación
        min_fit_clients=0,
        min_evaluate_clients=0,
        min_available_clients=0,
        on_fit_config_fn=on_fit_config_fn,
    )

    print("[Flower][Server] Iniciando servidor en", args.address)
    fl.server.start_server(
        server_address=args.address,
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
    )


if __name__ == "__main__":
    main()