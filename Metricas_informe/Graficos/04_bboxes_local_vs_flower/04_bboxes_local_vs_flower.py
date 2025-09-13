#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_bboxes_local_vs_flower.py
----------------------------
Comparativa de BoundingBoxes entre LOCAL y FLOWER.
Lee el consolidado y genera tablas + gráficos para method=BoundingBoxes
en los dos scopes: local vs flower. Incluye %slow_frames usando el P95
de latencia del grupo LOCAL como umbral de referencia.

Uso:
  python Metricas_informe/Graficos/04_bboxes_local_vs_flower/04_bboxes_local_vs_flower.py \
      --in Metricas_informe/raw/metrica_consolidada.csv \
      --out-dir Metricas_informe/Graficos/04_bboxes_local_vs_flower
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
    # Boxplot latencia para BB-Local y BB-Flower
    labels, data = [], []
    for s, g in df.groupby("scope"):
        label = "BB-Local" if str(s).lower()=="local" else "BB-Flower"
        labels.append(label)
        data.append(pd.to_numeric(g["latency_ms"], errors="coerce").dropna().values)
    if not data:
        return
    plt.figure()
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.title("Distribución de latencia (ms) - BoundingBoxes")
    plt.ylabel("latency_ms")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="BoundingBoxes: comparativa Local vs Flower con %slow_frames.")
    parser.add_argument("--in", "--input", dest="in_csv",
                        default=os.path.join("Metricas_informe","raw","metrica_consolidada.csv"),
                        help="Ruta del CSV consolidado (por defecto: Metricas_informe/raw/metrica_consolidada.csv)")
    parser.add_argument("--out-dir", "--output-dir", dest="out_dir",
                        default=os.path.join("Metricas_informe","report"),
                        help="Directorio de salida (por defecto: Metricas_informe/report)")
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

    # Filtro BoundingBoxes y scopes de interés
    for col in ["fps","latency_ms","cpu_percent","ram_mb"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[(df["method"]=="BoundingBoxes") & (df["scope"].str.lower().isin(["local","flower"]))]
    df["cpu_percent"] = df["cpu_percent"].clip(0,100)
    df = df[(df["fps"]>=0) & (df["latency_ms"]>=0) & (df["ram_mb"]>=0)]
    df = df.dropna(subset=["fps","latency_ms","cpu_percent","ram_mb","scope"])

    if df.empty:
        raise ValueError("No hay datos para BoundingBoxes en scopes local/flower.")

    # Resumen por scope
    rows = []
    for scope, g in df.groupby(df["scope"].str.lower(), sort=True):
        rec = {"scope": scope}
        rec.update(summarize_group(g))
        rows.append(rec)
    summary = pd.DataFrame(rows).sort_values("scope").reset_index(drop=True)

    # Eficiencia (FPS por 10% CPU)
    if not summary.empty:
        summary["efficiency_fps_per_10cpu"] = summary.apply(
            lambda r: (r["fps_mean"] / (r["cpu_percent_mean"]/10.0)) if r.get("cpu_percent_mean",0)>0 else 0.0,
            axis=1
        )

    # ---- % slow frames respecto al P95 de LOCAL ----
    local = df[df["scope"].str.lower()=="local"].copy()
    flower = df[df["scope"].str.lower()=="flower"].copy()

    local_p95 = pct(local["latency_ms"], 95) if not local.empty else np.nan

    def slow_frac(g, thr):
        if g.empty or np.isnan(thr):
            return np.nan
        s = pd.to_numeric(g["latency_ms"], errors="coerce").dropna()
        if s.size == 0:
            return np.nan
        return float((s > thr).sum()) / float(s.size) * 100.0

    slow_local = slow_frac(local, local_p95)
    slow_flower = slow_frac(flower, local_p95)

    # Añadir a summary
    summary["slow_frames_vs_localP95_%"] = summary["scope"].map(
        lambda s: slow_local if s=="local" else slow_flower
    )

    # Deltas (Flower vs Local)
    if set(summary["scope"]) == set(["flower","local"]):
        row_local  = summary[summary["scope"]=="local"].squeeze()
        row_flower = summary[summary["scope"]=="flower"].squeeze()
        metrics = ["fps_mean","latency_ms_mean","cpu_percent_mean","ram_mb_mean",
                   "efficiency_fps_per_10cpu","slow_frames_vs_localP95_%"]
        delta_rows = []
        for met in metrics:
            lv = float(row_local.get(met, np.nan))
            fv = float(row_flower.get(met, np.nan))
            delta_abs = fv - lv
            delta_pct = np.nan if (np.isnan(lv) or lv==0.0) else (delta_abs / lv * 100.0)
            delta_rows.append({"metric": met, "Flower_minus_Local": delta_abs, "delta_pct_vs_Local": delta_pct})
        deltas = pd.DataFrame(delta_rows)
    else:
        deltas = pd.DataFrame(columns=["metric","Flower_minus_Local","delta_pct_vs_Local"])

    # Guardar CSVs
    out_csv = out_dir / "bboxes_local_vs_flower.csv"
    summary.to_csv(out_csv, index=False)
    deltas.to_csv(out_dir / "bboxes_local_vs_flower_deltas.csv", index=False)

    # ---- Gráficos ----
    plot_df = summary.copy()
    plot_df["label"] = plot_df["scope"].map(lambda s: "BB-Local" if s=="local" else "BB-Flower")

    plot_bar(plot_df, "label", "fps_mean", "FPS medio - BoundingBoxes (Local vs Flower)", out_dir / "bb_fps_mean.png")
    plot_bar(plot_df, "label", "latency_ms_mean", "Latencia media (ms) - BoundingBoxes", out_dir / "bb_latency_mean.png")
    plot_bar(plot_df, "label", "efficiency_fps_per_10cpu", "Eficiencia (FPS por 10% CPU) - BoundingBoxes", out_dir / "bb_efficiency.png")
    plot_bar(plot_df, "label", "slow_frames_vs_localP95_%", "% frames lentos vs P95 Local - BoundingBoxes", out_dir / "bb_slow_frames.png")
    plot_box_latency(df, out_dir / "bb_latency_box.png")

    # Resumen Markdown
    summary_md = out_dir / "summary_bb_local_vs_flower.md"
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("# BoundingBoxes: Local vs Flower\n\n")
        f.write(f"- Dataset: {in_csv}\n")
        f.write(f"- Filas totales (BB local+flower): {len(df)}\n")
        f.write(f"- Umbral P95 local (latency_ms): {local_p95:.3f}\n\n")
        f.write("## Resumen por scope\n\n")
        f.write(summary.to_csv(index=False))
        f.write("\n## Deltas (Flower - Local)\n\n")
        f.write(deltas.to_csv(index=False))

    print(f"OK: escrito {out_csv} y figuras en {out_dir}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())