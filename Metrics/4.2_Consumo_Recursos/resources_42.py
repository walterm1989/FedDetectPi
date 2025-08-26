#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

DEFAULT_INPUT = "Metrics/out/section4_metrics_all.csv"
DEFAULT_OUT = "Metrics/4.2_Consumo_Recursos/figs"
DEFAULT_TABLES = "Metrics/4.2_Consumo_Recursos/tables"
DEFAULT_FORMAT = "png"
DEFAULT_DPI = 200
DEFAULT_RAM_TOTAL_MB = 4096

OUTPUT_FILENAMES = {
    "fig_cpu_barras": "fig_42_resumen_cpu_barras.{}",
    "fig_ram_barras": "fig_42_resumen_ram_barras.{}",
    "fig_cpu_box": "fig_42_box_cpu_pct.{}",
    "fig_ram_box": "fig_42_box_ram_mb.{}",
    "fig_cpu_hist": "fig_42_hist_cpu_pct.{}",
    "fig_ram_hist": "fig_42_hist_ram_mb.{}",
    "table_csv": "tabla_42_recursos_local.csv",
    "table_md": "tabla_42_recursos_local.md",
}

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

def parse_args():
    parser = argparse.ArgumentParser(description="Section 4.2 Resource Consumption Metrics")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT, help="Path to CSV input")
    parser.add_argument("--out", type=str, default=DEFAULT_OUT, help="Directory for output figures")
    parser.add_argument("--tables", type=str, default=DEFAULT_TABLES, help="Directory for output tables")
    parser.add_argument("--format", type=str, choices=["png", "pdf"], default=DEFAULT_FORMAT, help="Output format for figures")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI, help="DPI for figures")
    parser.add_argument("--only", type=str, default=None, help="Comma-separated list of methods to include")
    parser.add_argument("--ram-total-mb", type=int, default=DEFAULT_RAM_TOTAL_MB, help="Total RAM for margin calculation (MB)")
    parser.add_argument("--make-readme", action="store_true", help="Generate README_4.2.md")
    return parser.parse_args()

def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def load_data(input_path, only_methods=None):
    logging.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    # Clean up and robust numeric conversion
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="ignore")
    # Lowercase/strip column names for robustness
    df.columns = [c.strip() for c in df.columns]
    # Filter methods if needed
    if only_methods:
        methods = [m.strip() for m in only_methods.split(",")]
        df = df[df['method'].isin(methods)]
        logging.info(f"Filtered by --only: {methods}")
    method_counts = df['method'].value_counts().to_dict()
    for m, count in method_counts.items():
        logging.info(f"Count for method '{m}': {count}")
    return df

def compute_aggregates(df, ram_total_mb):
    # Ensure column names
    # 'method', 'cpu_pct', 'ram_mb'
    if 'cpu_pct' not in df.columns or 'ram_mb' not in df.columns or 'method' not in df.columns:
        raise ValueError("Input CSV must contain 'method', 'cpu_pct', and 'ram_mb' columns")
    groups = df.groupby('method')
    agg_list = []
    for method, group in groups:
        cpu_vals = pd.to_numeric(group['cpu_pct'], errors='coerce').dropna()
        ram_vals = pd.to_numeric(group['ram_mb'], errors='coerce').dropna()
        cpu_mean = cpu_vals.mean()
        cpu_p95 = np.percentile(cpu_vals, 95)
        cpu_max = cpu_vals.max()
        ram_mean = ram_vals.mean()
        ram_p95 = np.percentile(ram_vals, 95)
        ram_max = ram_vals.max()
        ram_free_margin_mean = ram_total_mb - ram_mean
        agg_list.append({
            "method": method,
            "cpu_mean": cpu_mean,
            "cpu_p95": cpu_p95,
            "cpu_max": cpu_max,
            "ram_mean": ram_mean,
            "ram_p95": ram_p95,
            "ram_max": ram_max,
            "ram_free_margin_mean": ram_free_margin_mean,
        })
    agg_df = pd.DataFrame(agg_list)
    agg_df = agg_df.sort_values(by=["cpu_mean", "ram_mean", "method"]).reset_index(drop=True)
    logging.info(f"Aggregates computed for {len(agg_df)} methods.")
    return agg_df

def make_note_resources(agg_df, cpu_mean_thr=80, cpu_p95_thr=90, ram_mean_thr=None, ram_p95_thr=None, ram_margin_thr=512):
    # Generates a dict: method -> note string about edge use & improvements
    notes = {}
    for _, row in agg_df.iterrows():
        method = row["method"]
        cpu_mean = row["cpu_mean"]
        cpu_p95 = row["cpu_p95"]
        ram_mean = row["ram_mean"]
        ram_p95 = row["ram_p95"]
        margin = row["ram_free_margin_mean"]
        issues = []
        if cpu_mean_thr is not None and cpu_mean > cpu_mean_thr:
            issues.append(f"CPU media >{cpu_mean_thr}%")
        if cpu_p95_thr is not None and cpu_p95 > cpu_p95_thr:
            issues.append(f"CPU P95 >{cpu_p95_thr}%")
        if ram_margin_thr is not None and margin < ram_margin_thr:
            issues.append(f"Margen RAM <{ram_margin_thr}MB")
        note = "OK" if not issues else "Mejorar: " + ", ".join(issues)
        notes[method] = note
        logging.info(f"Thresholds for {method}: {note}")
    return notes

def save_tables(agg_df, notes, out_dir):
    out_csv = os.path.join(out_dir, OUTPUT_FILENAMES["table_csv"])
    out_md = os.path.join(out_dir, OUTPUT_FILENAMES["table_md"])
    # Prepare display dataframe
    display_df = agg_df.copy()
    display_df["CPU media (%)"] = display_df["cpu_mean"].round(1)
    display_df["CPU P95 (%)"] = display_df["cpu_p95"].round(1)
    display_df["CPU pico (%)"] = display_df["cpu_max"].round(1)
    display_df["RAM media (MB)"] = display_df["ram_mean"].round(1)
    display_df["RAM P95 (MB)"] = display_df["ram_p95"].round(1)
    display_df["RAM pico (MB)"] = display_df["ram_max"].round(1)
    display_df["Margen libre medio (MB)"] = display_df["ram_free_margin_mean"].round(1)
    display_df["Uso de recursos en edge (y mejoras)"] = display_df["method"].map(notes)
    cols = [
        "method",
        "CPU media (%)",
        "CPU P95 (%)",
        "CPU pico (%)",
        "RAM media (MB)",
        "RAM P95 (MB)",
        "RAM pico (MB)",
        "Margen libre medio (MB)",
        "Uso de recursos en edge (y mejoras)"
    ]
    # Save CSV
    display_df[cols].to_csv(out_csv, index=False)
    # Save as Markdown
    table_md = tabulate(display_df[cols], headers="keys", tablefmt="github", showindex=False)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(table_md)
    logging.info(f"Table saved: {out_csv}")
    logging.info(f"Table saved: {out_md}")

def plot_bars_cpu(agg_df, out_dir, out_format, dpi):
    methods = agg_df["method"]
    cpu_mean = agg_df["cpu_mean"]
    cpu_p95 = agg_df["cpu_p95"]
    x = np.arange(len(methods))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(6, len(methods)*0.8), 5))
    bars1 = ax.bar(x - width/2, cpu_mean, width, label='Media')
    bars2 = ax.bar(x + width/2, cpu_p95, width, label='P95')
    for bar, val in zip(bars1, cpu_mean):
        ax.annotate(f"{val:.0f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0,3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, cpu_p95):
        ax.annotate(f"{val:.0f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0,3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    ax.set_ylabel("CPU (%)")
    ax.set_title("Consumo CPU por método")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    plt.tight_layout()
    fname = os.path.join(out_dir, OUTPUT_FILENAMES["fig_cpu_barras"].format(out_format))
    plt.savefig(fname, dpi=dpi, bbox_inches="tight")
    plt.close()
    logging.info(f"Figure saved: {fname}")

def plot_bars_ram(agg_df, out_dir, out_format, dpi):
    methods = agg_df["method"]
    ram_mean = agg_df["ram_mean"]
    ram_p95 = agg_df["ram_p95"]
    x = np.arange(len(methods))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(6, len(methods)*0.8), 5))
    bars1 = ax.bar(x - width/2, ram_mean, width, label='Media')
    bars2 = ax.bar(x + width/2, ram_p95, width, label='P95')
    for bar, val in zip(bars1, ram_mean):
        ax.annotate(f"{val:.0f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0,3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, ram_p95):
        ax.annotate(f"{val:.0f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0,3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    ax.set_ylabel("RAM (MB)")
    ax.set_title("Consumo RAM por método")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    plt.tight_layout()
    fname = os.path.join(out_dir, OUTPUT_FILENAMES["fig_ram_barras"].format(out_format))
    plt.savefig(fname, dpi=dpi, bbox_inches="tight")
    plt.close()
    logging.info(f"Figure saved: {fname}")

def plot_box_cpu(df, out_dir, out_format, dpi):
    methods = sorted(df['method'].unique())
    data = [df[df['method'] == m]['cpu_pct'].dropna().astype(float) for m in methods]
    fig, ax = plt.subplots(figsize=(max(6, len(methods)*0.8), 5))
    bp = ax.boxplot(data, patch_artist=True, labels=methods)
    ax.set_ylabel("CPU (%)")
    ax.set_title("Distribución CPU por método")
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    fname = os.path.join(out_dir, OUTPUT_FILENAMES["fig_cpu_box"].format(out_format))
    plt.savefig(fname, dpi=dpi, bbox_inches="tight")
    plt.close()
    logging.info(f"Figure saved: {fname}")

def plot_box_ram(df, out_dir, out_format, dpi):
    methods = sorted(df['method'].unique())
    data = [df[df['method'] == m]['ram_mb'].dropna().astype(float) for m in methods]
    fig, ax = plt.subplots(figsize=(max(6, len(methods)*0.8), 5))
    bp = ax.boxplot(data, patch_artist=True, labels=methods)
    ax.set_ylabel("RAM (MB)")
    ax.set_title("Distribución RAM por método")
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    fname = os.path.join(out_dir, OUTPUT_FILENAMES["fig_ram_box"].format(out_format))
    plt.savefig(fname, dpi=dpi, bbox_inches="tight")
    plt.close()
    logging.info(f"Figure saved: {fname}")

def plot_hist_cpu(df, out_dir, out_format, dpi):
    plt.figure(figsize=(8,5))
    for method in sorted(df['method'].unique()):
        vals = df[df['method'] == method]['cpu_pct'].dropna().astype(float)
        plt.hist(vals, bins=20, alpha=0.5, label=method)
    plt.xlabel("CPU (%)")
    plt.ylabel("Frecuencia")
    plt.title("Histograma CPU (%) por método")
    plt.grid(axis='y', linestyle=':', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(out_dir, OUTPUT_FILENAMES["fig_cpu_hist"].format(out_format))
    plt.savefig(fname, dpi=dpi, bbox_inches="tight")
    plt.close()
    logging.info(f"Figure saved: {fname}")

def plot_hist_ram(df, out_dir, out_format, dpi):
    plt.figure(figsize=(8,5))
    for method in sorted(df['method'].unique()):
        vals = df[df['method'] == method]['ram_mb'].dropna().astype(float)
        plt.hist(vals, bins=20, alpha=0.5, label=method)
    plt.xlabel("RAM (MB)")
    plt.ylabel("Frecuencia")
    plt.title("Histograma RAM (MB) por método")
    plt.grid(axis='y', linestyle=':', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(out_dir, OUTPUT_FILENAMES["fig_ram_hist"].format(out_format))
    plt.savefig(fname, dpi=dpi, bbox_inches="tight")
    plt.close()
    logging.info(f"Figure saved: {fname}")

def write_readme(out_dir, figs_dir, tables_dir, out_format, agg_df, notes):
    readme_path = os.path.join(out_dir, "README_4.2.md")
    rel_figs = lambda fname: os.path.relpath(os.path.join(figs_dir, fname), out_dir)
    rel_tables = lambda fname: os.path.relpath(os.path.join(tables_dir, fname), out_dir)
    desc = (
        "# Sección 4.2: Consumo de Recursos\n\n"
        "Este análisis contiene métricas de consumo de CPU y RAM para cada método evaluado en el edge.\n"
        "Se incluyen medias, percentiles, máximos, márgenes de memoria y notas de uso recomendadas.\n"
        "\n"
        "## Figuras\n"
        f"- [Resumen CPU (barras)]({rel_figs(OUTPUT_FILENAMES['fig_cpu_barras'].format(out_format))})\n"
        f"- [Resumen RAM (barras)]({rel_figs(OUTPUT_FILENAMES['fig_ram_barras'].format(out_format))})\n"
        f"- [Boxplot CPU (%)]({rel_figs(OUTPUT_FILENAMES['fig_cpu_box'].format(out_format))})\n"
        f"- [Boxplot RAM (MB)]({rel_figs(OUTPUT_FILENAMES['fig_ram_box'].format(out_format))})\n"
        f"- [Histograma CPU (%)]({rel_figs(OUTPUT_FILENAMES['fig_cpu_hist'].format(out_format))})\n"
        f"- [Histograma RAM (MB)]({rel_figs(OUTPUT_FILENAMES['fig_ram_hist'].format(out_format))})\n"
        "\n"
        "## Tablas\n"
        f"- [Tabla resumen CSV]({rel_tables(OUTPUT_FILENAMES['table_csv'])})\n"
        f"- [Tabla resumen Markdown]({rel_tables(OUTPUT_FILENAMES['table_md'])})\n"
        "\n"
        "## Notas sobre umbrales y recomendaciones\n"
        "- CPU media &gt;80% o P95 &gt;90%: Mejorar eficiencia.\n"
        "- Margen libre RAM &lt;512MB: Riesgo, puede requerir optimización.\n"
        "\n"
        "## Detalle de métodos\n"
    )
    # Add a table of methods and their notes
    method_notes = "\n| Método | Nota |\n|---|---|\n"
    for _, row in agg_df.iterrows():
        m = row["method"]
        method_notes += f"| {m} | {notes[m]} |\n"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(desc + method_notes)
    logging.info(f"README generated: {readme_path}")

def main():
    setup_logger()
    args = parse_args()

    ensure_dirs(args.out, args.tables)

    df = load_data(args.input, args.only)
    agg_df = compute_aggregates(df, args.ram_total_mb)
    notes = make_note_resources(agg_df)
    save_tables(agg_df, notes, args.tables)
    plot_bars_cpu(agg_df, args.out, args.format, args.dpi)
    plot_bars_ram(agg_df, args.out, args.format, args.dpi)
    plot_box_cpu(df, args.out, args.format, args.dpi)
    plot_box_ram(df, args.out, args.format, args.dpi)
    plot_hist_cpu(df, args.out, args.format, args.dpi)
    plot_hist_ram(df, args.out, args.format, args.dpi)

    if args.make_readme:
        write_readme(
            out_dir=os.path.dirname(os.path.abspath(args.out)),
            figs_dir=args.out,
            tables_dir=args.tables,
            out_format=args.format,
            agg_df=agg_df,
            notes=notes
        )
    logging.info("Completed Section 4.2 resource metrics.")

if __name__ == "__main__":
    main()