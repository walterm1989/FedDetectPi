#!/usr/bin/env python3

import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from collections import OrderedDict

def parse_args():
    parser = argparse.ArgumentParser(
        description="Section 4.1 Latency Metrics: aggregation, plots, tables, and README."
    )
    parser.add_argument("--input", default="Metrics/out/section4_metrics_all.csv",
                        help="Input CSV with metrics (default: %(default)s)")
    parser.add_argument("--out", default="Metrics/4.1_Latencia/figs",
                        help="Directory for output figures (default: %(default)s)")
    parser.add_argument("--tables", default="Metrics/4.1_Latencia/tables",
                        help="Directory for output tables (default: %(default)s)")
    parser.add_argument("--format", default="png", choices=["png", "pdf"],
                        help="Figure file format (default: %(default)s)")
    parser.add_argument("--dpi", type=int, default=200,
                        help="Figure DPI (default: %(default)s)")
    parser.add_argument("--only", default=None,
                        help="Comma-separated list of methods to include (default: all)")
    parser.add_argument("--rt-fps", type=float, default=6,
                        help="Real-time FPS threshold (default: %(default)s)")
    parser.add_argument("--rt-p95-lat-ms", type=float, default=250,
                        help="Real-time P95 latency (ms) threshold (default: %(default)s)")
    parser.add_argument("--make-readme", action="store_true",
                        help="If set, generate README_4.1.md")
    parser.add_argument("--reference-input", default=None,
                        help="Optional reference metrics CSV path")
    return parser.parse_args()

# Mappings for method labels to Esquema
LOCAL_LABELS = {
    "KeyPoints-ResNet50": "Keypoint R-CNN (RPi – local)",
    "BBoxes-YOLOv4tiny": "Bounding Boxes (RPi – local)",
    "FlowerAI": "Flower AI (RPi – local)"
}
REFERENCE_SUFFIX = " (Portátil – referencia)"

def ensure_dirs(*paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)

def load_data(path, only=None):
    df = pd.read_csv(path)
    # Parse numeric columns (allow for both ',' and '.' decimal separators)
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = pd.to_numeric(df[col].str.replace(",", ".", regex=False), errors="ignore")
            except Exception:
                pass
    # Standardize method column name
    if "method" not in df.columns:
        # Try alternatives
        for c in df.columns:
            if c.lower() in ("esquema", "model", "method_name"):
                df = df.rename(columns={c: "method"})
                break
    if only:
        allowed = [x.strip() for x in only.split(",")]
        df = df[df["method"].isin(allowed)]
    return df

def aggregate_latency(df):
    groups = df.groupby("method")
    result = []
    for method, group in groups:
        fps_mean = group["fps_inst"].mean()
        latency_p95_ms = np.percentile(group["latency_ms"], 95)
        var_latency_ms = group["latency_ms"].var(ddof=0)
        result.append({
            "method": method,
            "fps_mean": fps_mean,
            "latency_p95_ms": latency_p95_ms,
            "var_latency_ms": var_latency_ms
        })
    return pd.DataFrame(result)

def method_to_esquema(method, is_reference=False):
    if not is_reference:
        label = LOCAL_LABELS.get(method, method + " (RPi – local)")
    else:
        # For reference, try to map to local label, but with suffix
        base_label = LOCAL_LABELS.get(method, method)
        label = f"{base_label}{REFERENCE_SUFFIX}"
    return label

def make_note(method, fps_mean, p95_ms, rt_fps, rt_p95):
    # Tier A
    if pd.notna(fps_mean) and pd.notna(p95_ms):
        if fps_mean >= rt_fps and p95_ms <= rt_p95:
            note = ("Apto para flujo continuo en edge. Mantener resolución actual o reducir si se requiere margen; "
                    "considerar pipeline asíncrono para estabilidad.")
        # Tier B
        elif fps_mean >= rt_fps/2 or p95_ms <= 2*rt_p95:
            if "Flower" in method:
                note = ("Mejor desempeño relativo en FPS dentro de RPi. Adecuado para conteo/presencia a baja frecuencia; "
                        "optimizable con resolución reducida y pipeline asíncrono.")
            else:
                note = ("Útil para monitorización básica o por eventos. Mejorable con menor resolución (p. ej., 320×240), "
                        "pipeline asíncrono (captura/inferencia en hilos), ejecución headless y ajuste de umbral/NMS.")
        # Tier C
        else:
            note = ("Referencia de mayor complejidad; recomendable para análisis offline o como línea base. "
                    "Para uso en vivo en RPi, optar por modelos más ligeros o esquema por etapas (caja → pose).")
    else:
        # If any of the metrics is nan, default to Tier C
        note = ("Referencia de mayor complejidad; recomendable para análisis offline o como línea base. "
                "Para uso en vivo en RPi, optar por modelos más ligeros o esquema por etapas (caja → pose).")

    # Special cases
    if pd.isna(p95_ms):
        note += " P95 no disponible en esta corrida; la distribución sugiere latencia en centenas de ms según histogramas."
    # Suggestions by method
    if method == "KeyPoints-ResNet50":
        note += " Explorar variantes lightweight (BlazePose/MoveNet) o pipeline por etapas."
    if "BBoxes-YOLOv4tiny" in method:
        note += " Ajustar resolución/stride y NMS; priorizar GPU/NPU si estuviera disponible."
    if "FlowerAI" in method:
        note += " Optimizar ciclo local (captura/cola) y tamaño de lote si aplica."
    return note

def make_tables(df_agg, tables_dir, rt_fps, rt_p95_lat_ms, is_reference=False):
    table_csv = os.path.join(tables_dir, f"tabla_41_latencia_{'referencia' if is_reference else 'local'}.csv")
    table_md = table_csv.replace(".csv", ".md")
    rows = []
    files_written = []
    for _, row in df_agg.iterrows():
        esquema = method_to_esquema(row["method"], is_reference)
        fps = row["fps_mean"]
        p95 = row["latency_p95_ms"]
        var = row["var_latency_ms"]

        note = make_note(row["method"], fps, p95, rt_fps, rt_p95_lat_ms)

        rows.append(OrderedDict([
            ("Esquema", esquema),
            ("Media FPS", f"{fps:.2f}"),
            ("P95 (ms)", f"{p95:.1f}"),
            ("Varianza (lat_ms)", f"{var:.1f}"),
            ("Aplicabilidad en edge (y mejoras)", note)
        ]))
    df_table = pd.DataFrame(rows)
    df_table.to_csv(table_csv, index=False)
    files_written.append(table_csv)
    with open(table_md, "w", encoding="utf-8") as fmd:
        fmd.write(tabulate(df_table, headers="keys", tablefmt="github", showindex=False))
        fmd.write("\n")
    files_written.append(table_md)
    return files_written

def plot_box(df, out_dir, fmt, dpi):
    fig, ax = plt.subplots(figsize=(7, 4))
    methods = df["method"].unique()
    data = [df[df["method"] == m]["latency_ms"] for m in methods]
    ax.boxplot(data, labels=[method_to_esquema(m) for m in methods], patch_artist=True)
    ax.set_ylabel("Latencia (ms)")
    ax.set_title("Distribución de latencias por esquema")
    ax.grid(True, axis='y')
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    fname = os.path.join(out_dir, f"fig_41_box_latency_ms.{fmt}")
    fig.savefig(fname, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return fname

def plot_hist(df, out_dir, fmt, dpi):
    files = []
    # Combined histogram (log x)
    fig, ax = plt.subplots(figsize=(7, 4))
    for m in df["method"].unique():
        vals = df[df["method"] == m]["latency_ms"]
        ax.hist(vals, bins=40, alpha=0.6, label=method_to_esquema(m), histtype="stepfilled")
    ax.set_xlabel("Latencia (ms)")
    ax.set_ylabel("Frecuencia")
    ax.set_xscale("log")
    ax.set_title("Histograma de latencia (log escala)")
    ax.grid(True, axis='y')
    ax.legend()
    plt.tight_layout()
    fname_combined = os.path.join(out_dir, f"fig_41_hist_latency_ms.{fmt}")
    fig.savefig(fname_combined, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    files.append(fname_combined)
    # Per-method histograms
    for m in df["method"].unique():
        vals = df[df["method"] == m]["latency_ms"]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(vals, bins=40, alpha=0.75, color="#5b9bd5")
        ax.set_xlabel("Latencia (ms)")
        ax.set_ylabel("Frecuencia")
        ax.set_xscale("log")
        ax.set_title(f"Histograma: {method_to_esquema(m)}")
        ax.grid(True, axis='y')
        plt.tight_layout()
        fname = os.path.join(out_dir, f"fig_41_hist_latency_ms_{m}.{fmt}")
        fig.savefig(fname, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        files.append(fname)
    return files

def plot_summary_bars(df_agg, out_dir, fmt, dpi):
    files = []
    # p95 latency
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ylabels = [method_to_esquema(m) for m in df_agg["method"]]
    vals = df_agg["latency_p95_ms"]
    bars = ax.barh(ylabels, vals, color="#ed7d31", alpha=0.8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f"{v:.1f} ms", va="center", ha="left", fontsize=10)
    ax.set_xlabel("Latencia P95 (ms)")
    ax.set_title("Resumen de latencia P95 por esquema")
    ax.grid(True, axis="x")
    plt.tight_layout()
    fname_p95 = os.path.join(out_dir, f"fig_41_resumen_latency_p95.{fmt}")
    fig.savefig(fname_p95, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    files.append(fname_p95)
    # mean FPS
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ylabels = [method_to_esquema(m) for m in df_agg["method"]]
    vals = df_agg["fps_mean"]
    bars = ax.barh(ylabels, vals, color="#4472c4", alpha=0.8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f"{v:.2f} FPS", va="center", ha="left", fontsize=10)
    ax.set_xlabel("Media FPS")
    ax.set_title("Resumen de media FPS por esquema")
    ax.grid(True, axis="x")
    plt.tight_layout()
    fname_fps = os.path.join(out_dir, f"fig_41_resumen_fps_mean.{fmt}")
    fig.savefig(fname_fps, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    files.append(fname_fps)
    return files

def write_readme(fig_files, table_files, outdir, tablesdir, args):
    md = []
    md.append("# Sección 4.1: Latencia de esquemas de inferencia en dispositivos Edge\n")
    md.append("Este informe resume los resultados de latencia y rendimiento para diferentes esquemas de inferencia ejecutados localmente en Raspberry Pi y, opcionalmente, en referencia portátil.\n")
    md.append("## Figuras")
    for f in fig_files:
        relpath = os.path.relpath(f, os.path.dirname(__file__))
        md.append(f"- ![]({relpath})")
    md.append("\n## Tablas")
    md.append("\n**Nota:** En todas las tablas, la columna \"Aplicabilidad en edge (y mejoras)\" indica la idoneidad para uso en edge y posibles optimizaciones específicas por modelo.\n")
    for f in table_files:
        if f.endswith(".md"):
            relpath = os.path.relpath(f, os.path.dirname(__file__))
            md.append(f"- [{os.path.basename(f)}]({relpath})")
    out_path = os.path.join(os.path.dirname(__file__), "README_4.1.md")
    with open(out_path, "w", encoding="utf-8") as fmd:
        fmd.write("\n".join(md))
    print(f"README written: {out_path}")

def main():
    args = parse_args()
    ensure_dirs(args.out, args.tables)
    print(f"Ensured output dirs: {args.out}, {args.tables}")

    # Load local data
    print(f"Loading data from: {args.input}")
    df = load_data(args.input, only=args.only)
    print(f"Loaded {len(df)} rows from local metrics.")

    # Aggregate local
    df_agg = aggregate_latency(df)
    print(f"Aggregated {len(df_agg)} local methods: {list(df_agg['method'])}")

    # Tables local
    table_files = make_tables(df_agg, args.tables, args.rt_fps, args.rt_p95_lat_ms, is_reference=False)
    print(f"Generated local tables: {table_files}")

    fig_files = []
    # Plots local (all methods together)
    fig_files.append(plot_box(df, args.out, args.format, args.dpi))
    fig_files += plot_hist(df, args.out, args.format, args.dpi)
    fig_files += plot_summary_bars(df_agg, args.out, args.format, args.dpi)
    print(f"Generated local figures: {fig_files}")

    # Reference data (optional)
    if args.reference_input:
        print(f"Loading reference data: {args.reference_input}")
        df_ref = load_data(args.reference_input, only=args.only)
        print(f"Loaded {len(df_ref)} rows from reference metrics.")
        df_agg_ref = aggregate_latency(df_ref)
        print(f"Aggregated {len(df_agg_ref)} reference methods: {list(df_agg_ref['method'])}")
        ref_table_files = make_tables(df_agg_ref, args.tables, args.rt_fps, args.rt_p95_lat_ms, is_reference=True)
        print(f"Generated reference tables: {ref_table_files}")
        table_files += ref_table_files

    # README
    if args.make_readme:
        write_readme(fig_files, table_files, args.out, args.tables, args)

    print("All done.")
    print(f"Processed methods: {list(df_agg['method'])}")
    print(f"Thresholds applied: real-time FPS={args.rt_fps}, P95 latency={args.rt_p95_lat_ms} ms")
    print("Output files:")
    for f in fig_files + table_files:
        print(" -", f)

if __name__ == "__main__":
    main()