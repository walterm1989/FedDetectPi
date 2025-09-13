#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_comparativa_flower.py
------------------------
Comparativa en FLOWER entre BoundingBoxes y KeyPoints.
Lee el consolidado y genera tablas + gráficos solo con scope=flower.

Uso:
  python Metricas_informe/report/03_comparativa_flower.py \
      --in Metricas_informe/raw/metrica_consolidada.csv \
      --out-dir Metricas_informe/report
"""

import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REQUIRED_COLS = [
    "timestamp","fps","latency_ms","cpu_percent","ram_mb",
    "method","model","scope","file_source","frame_idx"
]

def pct(series: pd.Series, p: float) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna().values
    return float(np.percentile(s, p)) if s.size else np.nan

def summarize_group(gdf: pd.DataFrame) -> dict:
    out = {}
    for met in ["fps","latency_ms"]:
        s = pd.to_numeric(gdf[met], errors="coerce").dropna()
        out[f"{met}_mean"] = float(s.mean()) if s.size else np.nan
        out[f"{met}_p50"]  = pct(s, 50)
        out[f"{met}_p90"]  = pct(s, 90)
        out[f"{met}_p95"]  = pct(s, 95)
        out[f"{met}_p99"]  = pct(s, 99)
        out[f"{met}_std"]  = float(s.std(ddof=1)) if s.size>1 else 0.0
        out[f"{met}_min"]  = float(s.min()) if s.size else np.nan
        out[f"{met}_max"]  = float(s.max()) if s.size else np.nan
    for met in ["cpu_percent","ram_mb"]:
        s = pd.to_numeric(gdf[met], errors="coerce").dropna()
        out[f"{met}_mean"] = float(s.mean()) if s.size else np.nan
        out[f"{met}_p95"]  = pct(s, 95)
    out["n"] = int(len(gdf))
    return out

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def plot_bar(df_src: pd.DataFrame, x_col: str, y_col: str, title: str, out_file: Path):
    plt.figure()
    plt.bar(df_src[x_col].astype(str), df_src[y_col])
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()

def plot_box_latency(df: pd.DataFrame, out_file: Path):
    # Boxplot de latencia para BB-Flower y KP-Flower
    labels, data = [], []
    for m, g in df.groupby("method"):
        labels.append("BB-Flower" if "BoundingBoxes" in str(m) else "KP-Flower")
        data.append(pd.to_numeric(g["latency_ms"], errors="coerce").dropna().values)
    if not data:
        return
    plt.figure()
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.title("Distribución de latencia (ms) - Flower")
    plt.ylabel("latency_ms")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Comparativa FLOWER: BoundingBoxes vs KeyPoints.")
    # Alias extra para evitar líos en PowerShell
    parser.add_argument("--in", "--input", dest="in_csv",
                        default=os.path.join("Metricas_informe","raw","metrica_consolidada.csv"),
                        help="Ruta del CSV consolidado (por defecto: Metricas_informe/raw/metrica_consolidada.csv)")
    parser.add_argument("--out-dir", "--output-dir", dest="out_dir",
                        default=os.path.join("Metricas_informe","report"),
                        help="Directorio de salida para CSVs/figuras (por defecto: Metricas_informe/report)")
    args = parser.parse_args()

    in_csv = Path(args.in_csv)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    if not in_csv.exists():
        raise FileNotFoundError(f"No existe el archivo de entrada: {in_csv}")

    df = pd.read_csv(in_csv)

    # Validaciones
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas en el consolidado: {missing}")

    # Filtro FLOWER y solo métodos de interés
    for col in ["fps","latency_ms","cpu_percent","ram_mb"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[(df["scope"].str.lower()=="flower") & (df["method"].isin(["BoundingBoxes","KeyPoints"]))]
    df["cpu_percent"] = df["cpu_percent"].clip(0,100)
    df = df[(df["fps"]>=0) & (df["latency_ms"]>=0) & (df["ram_mb"]>=0)]
    df = df.dropna(subset=["fps","latency_ms","cpu_percent","ram_mb","method"])

    if df.empty:
        raise ValueError("No hay datos para scope=flower y métodos BoundingBoxes/KeyPoints.")

    # Resumen por método
    rows = []
    for method, g in df.groupby("method", sort=True):
        rec = {"method": method}
        rec.update(summarize_group(g))
        rows.append(rec)
    summary = pd.DataFrame(rows).sort_values("method").reset_index(drop=True)

    # Eficiencia (FPS por 10% CPU)
    if not summary.empty:
        summary["efficiency_fps_per_10cpu"] = summary.apply(
            lambda r: (r["fps_mean"] / (r["cpu_percent_mean"]/10.0)) if r["cpu_percent_mean"] and r["cpu_percent_mean"]>0 else 0.0,
            axis=1
        )

    # Deltas (KeyPoints vs BoundingBoxes)
    def extract_row(m):
        return summary.loc[summary["method"]==m].squeeze()

    if set(summary["method"]) == set(["BoundingBoxes","KeyPoints"]):
        bb = extract_row("BoundingBoxes")
        kp = extract_row("KeyPoints")
        metrics_for_delta = ["fps_mean","latency_ms_mean","cpu_percent_mean","ram_mb_mean","efficiency_fps_per_10cpu"]
        delta_rows = []
        for met in metrics_for_delta:
            bbv = float(bb.get(met, np.nan))
            kpv = float(kp.get(met, np.nan))
            if np.isnan(bbv) or bbv == 0.0:
                delta_pct = np.nan
            else:
                delta_pct = (kpv - bbv) / bbv * 100.0
            delta_rows.append({"metric": met, "KP_minus_BB": kpv - bbv, "delta_pct_vs_BB": delta_pct})
        deltas = pd.DataFrame(delta_rows)
    else:
        deltas = pd.DataFrame(columns=["metric","KP_minus_BB","delta_pct_vs_BB"])

    # Guardar CSVs
    out_csv = out_dir / "flower_comparison.csv"
    summary.to_csv(out_csv, index=False)
    deltas.to_csv(out_dir / "flower_comparison_deltas.csv", index=False)

    # Gráficos (barras y boxplot)
    plot_df = summary.copy()
    plot_df["label"] = plot_df["method"].map(lambda m: "BB-Flower" if "BoundingBoxes" in str(m) else "KP-Flower")

    plot_bar(plot_df, "label", "fps_mean", "FPS medio - Flower", out_dir / "flower_fps_mean.png")
    plot_bar(plot_df, "label", "latency_ms_mean", "Latencia media (ms) - Flower", out_dir / "flower_latency_mean.png")
    plot_bar(plot_df, "label", "efficiency_fps_per_10cpu", "Eficiencia (FPS por 10% CPU) - Flower", out_dir / "flower_efficiency.png")
    plot_box_latency(df, out_dir / "flower_latency_box.png")

    # Resumen Markdown
    summary_md = out_dir / "summary_flower.md"
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("# Comparativa FLOWER (BoundingBoxes vs KeyPoints)\n\n")
        f.write(f"- Dataset: {in_csv}\n")
        f.write(f"- Filas totales (FLOWER): {len(df)}\n\n")
        f.write("## Resumen por método\n\n")
        f.write(summary.to_csv(index=False))
        f.write("\n## Deltas (KeyPoints - BoundingBoxes)\n\n")
        f.write(deltas.to_csv(index=False))

    print(f"OK: escrito {out_csv} y figuras en {out_dir}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())