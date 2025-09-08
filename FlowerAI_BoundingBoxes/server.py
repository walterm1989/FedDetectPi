#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Servidor Flower AI para plano de control de parámetros de detección HOG+SVM.
No se realiza entrenamiento; únicamente se envía configuración a los clientes
en cada ronda (threshold, win_stride, padding, scale).
"""

import argparse
from typing import Dict

import flwr as fl


def build_on_fit_config_fn(args: argparse.Namespace):
    """Devuelve una función que genera la config enviada a cada ronda."""

    def on_fit_config_fn(server_round: int) -> Dict[str, float]:
        # Se pueden ajustar dinámicamente por ronda si se desea
        return {
            "threshold": float(args.threshold),
            "win_stride": int(args.win_stride),
            "padding": int(args.padding),
            "scale": float(args.scale),
        }

    return on_fit_config_fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Servidor Flower (plano de control HOG+SVM)")
    parser.add_argument(
        "--address",
        type=str,
        default="0.0.0.0:8080",
        help="Dirección del servidor (host:port), por defecto 0.0.0.0:8080",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=9999,
        help="Número de rondas (usar un valor alto como 'infinito'), por defecto 9999",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Umbral para filtrar detecciones por peso (weight >= threshold)",
    )
    parser.add_argument(
        "--win-stride",
        type=int,
        default=8,
        help="winStride (píxeles) para HOG detectMultiScale",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=8,
        help="padding (píxeles) para HOG detectMultiScale",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.05,
        help="Factor de escala para HOG detectMultiScale",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    strategy = fl.server.strategy.FedAvg(
        on_fit_config_fn=build_on_fit_config_fn(args),
    )

    print(
        f"Iniciando servidor Flower en {args.address} | rondas={args.rounds} | "
        f"threshold={args.threshold} | win_stride={args.win_stride} | "
        f"padding={args.padding} | scale={args.scale}"
    )

    fl.server.start_server(
        server_address=args.address,
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
    )


if __name__ == "__main__":
    main()