#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_metricas_totales.py
----------------------
Métricas totales con los 4 archivos (grupos por method/scope).
Lee el consolidado y genera tablas y gráficos de resumen.

Uso:
    python Metricas_informe/report/01_metricas_totales.py \
        --in Metricas_informe/raw/metrica_consolidada.csv \
        --out-dir Metricas_informe/report
"""

import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


REQUIRED_COLS = ["timestamp","fps","latency_ms","cpu_percent","ram_mb","method","model","scope","file_source","frame_idx"]


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
    # Boxplot de latencia por grupo (method/scope)
    labels = []
    data = []
    for (m, s), g in df.groupby(["method","scope"]):
        labels.append( ("BB" if "BoundingBoxes" in str(m) else "KP") + "-" + str(s).capitalize() )
        data.append( pd.to_numeric(g["latency_ms"], errors="coerce").dropna().values )
    plt.figure()
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.title("Distribución de latencia (ms) por grupo")
    plt.ylabel("latency_ms")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Métricas totales con los 4 archivos (grupos por method/scope).")
    parser.add_argument("--in", dest="in_csv", default=os.path.join("Metricas_informe","raw","metrica_consolidada.csv"),
                        help="Ruta del CSV consolidado (por defecto: Metricas_informe/raw/metrica_consolidada.csv)")
    parser.add_argument("--out-dir", dest="out_dir", default=os.path.join("Metricas_informe","report"),
                        help="Directorio de salida para CSVs/figuras (por defecto: Metricas_informe/report)")
    args = parser.parse_args()

    in_csv = Path(args.in_csv)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    if not in_csv.exists():
        raise FileNotFoundError(f"No existe el archivo de entrada: {in_csv}")

    df = pd.read_csv(in_csv)

    # Validaciones básicas
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas en el consolidado: {missing}")

    # Limpiar tipos/rangos
    for col in ["fps","latency_ms","cpu_percent","ram_mb"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["cpu_percent"] = df["cpu_percent"].clip(0,100)
    df = df[(df["fps"]>=0) & (df["latency_ms"]>=0) & (df["ram_mb"]>=0)]
    df = df.dropna(subset=["fps","latency_ms","cpu_percent","ram_mb","method","scope"])

    # Resumen por grupo method/scope
    rows = []
    for (method, scope), g in df.groupby(["method","scope"], sort=True):
        rec = {"method": method, "scope": scope}
        rec.update(summarize_group(g))
        rows.append(rec)
    overall = pd.DataFrame(rows).sort_values(["method","scope"]).reset_index(drop=True)

    # Eficiencia (FPS por 10% CPU)
    if not overall.empty:
        overall["efficiency_fps_per_10cpu"] = overall.apply(
            lambda r: (r["fps_mean"] / (r["cpu_percent_mean"]/10.0)) if r["cpu_percent_mean"] and r["cpu_percent_mean"]>0 else 0.0,
            axis=1
        )

    # Guardar CSVs
    overall_csv = out_dir / "overall_summary.csv"
    overall.to_csv(overall_csv, index=False)

    # Gráficos
    plot_df = overall.copy()
    plot_df["label"] = plot_df.apply(lambda r: f"{'BB' if 'BoundingBoxes' in str(r['method']) else 'KP'}-{str(r['scope']).capitalize()}", axis=1)

    plot_bar(plot_df, "label", "fps_mean", "FPS medio por grupo", out_dir / "overall_fps_mean.png")
    plot_bar(plot_df, "label", "latency_ms_mean", "Latencia media (ms) por grupo", out_dir / "overall_latency_mean.png")
    plot_bar(plot_df, "label", "efficiency_fps_per_10cpu", "Eficiencia (FPS por 10% CPU)", out_dir / "overall_efficiency.png")
    plot_box_latency(df, out_dir / "overall_latency_box.png")

    # Resumen en Markdown
    summary_md = out_dir / "summary_totales.md"
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("# Métricas totales (4 archivos)\n\n")
        f.write(f"- Dataset: {in_csv}\n")
        f.write(f"- Filas totales: {len(df)}\n\n")
        f.write("## Resumen por grupo (method/scope)\n\n")
        f.write(overall.to_csv(index=False))
        if not overall.empty:
            best_fps = overall.sort_values("fps_mean", ascending=False).iloc[0]
            best_lat = overall.sort_values("latency_ms_mean").iloc[0]
            best_eff = overall.sort_values("efficiency_fps_per_10cpu", ascending=False).iloc[0]
            f.write("\n## Hallazgos automáticos\n")
            f.write(f"- Mayor FPS medio: {best_fps['method']} / {best_fps['scope']} ({best_fps['fps_mean']:.2f} FPS)\n")
            f.write(f"- Menor latencia media: {best_lat['method']} / {best_lat['scope']} ({best_lat['latency_ms_mean']:.2f} ms)\n")
            f.write(f"- Mayor eficiencia (FPS por 10% CPU): {best_eff['method']} / {best_eff['scope']} ({best_eff['efficiency_fps_per_10cpu']:.2f})\n")

    print(f"OK: escrito {overall_csv} y figuras en {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
