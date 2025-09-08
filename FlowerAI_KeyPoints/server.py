#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Servidor Flower (plano de control) para KeyPoints (Keypoint R-CNN ResNet50-FPN)
# Publica:
#  - conf_thr (float, default 0.5)
#  - input_size (int, default 640)
#  - max_frames (int, opcional; 0 => sin límite)
#  - draw (int bool 0/1)

import argparse
from typing import Dict

import flwr as fl


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Servidor Flower - KeyPoints (Keypoint R-CNN ResNet50-FPN)")
    p.add_argument("--address", type=str, default="0.0.0.0:8080", help="Dirección host:puerto del servidor Flower")
    p.add_argument("--rounds", type=int, default=1, help="Número de rondas (placeholder)")
    p.add_argument("--conf", type=float, default=0.5, help="Umbral de confianza")
    p.add_argument("--input-size", type=int, default=640, help="Lado corto de la imagen de entrada")
    p.add_argument("--max-frames", type=int, default=0, help="Límite de frames a procesar (0 = sin límite)")
    p.add_argument("--draw", type=int, choices=[0, 1], default=0, help="Dibujar skeleton en el cliente (0/1)")
    return p


def make_on_fit_config_fn(conf_thr: float, input_size: int, max_frames: int, draw: int):
    def on_fit_config_fn(server_round: int) -> Dict[str, int]:
        return {
            "conf_thr": float(conf_thr),
            "input_size": int(input_size),
            "max_frames": int(max_frames),
            "draw": int(draw),
            "server_round": int(server_round),
        }

    return on_fit_config_fn


def main() -> None:
    args = build_argparser().parse_args()

    on_fit_config_fn = make_on_fit_config_fn(args.conf, args.input_size, args.max_frames, args.draw)

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.0,
        fraction_evaluate=0.0,
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