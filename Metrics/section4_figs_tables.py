#!/usr/bin/env python3

import argparse
import os
import sys
import warnings
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Utility Functions ---

def load_data(input_path, only_methods=None):
    df = pd.read_csv(input_path)
    df.columns = [col.strip() for col in df.columns]

    # Ensure numeric columns (excluding 'Method', 'Frame', etc.)
    non_numeric = ['Method', 'Frame', 'Timestamp', 'Video', 'Extra']
    for col in df.columns:
        if col not in non_numeric:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if only_methods:
        df = df[df['Method'].isin(only_methods)]
    filtered_methods = df['Method'].unique()
    if len(filtered_methods) == 0:
        print("ERROR: After filtering, no methods remain. Exiting.", file=sys.stderr)
        sys.exit(1)
    return df

def compute_aggregates(df):
    result = []
    method_groups = df.groupby('Method')
    for method, group in method_groups:
        entry = {'Method': method}
        # Count frames
        entry['n_frames'] = group.shape[0]

        # FPS statistics
        fps = group['fps_inst'] if 'fps_inst' in group else pd.Series(dtype=float)
        for stat, func in [('fps_mean', np.nanmean), ('fps_p50', lambda x: np.nanpercentile(x, 50)),
                           ('fps_p95', lambda x: np.nanpercentile(x, 95))]:
            entry[stat] = safe_stat(fps, func)

        # Latency statistics (ms)
        latency = group['latency_ms'] if 'latency_ms' in group else pd.Series(dtype=float)
        for stat, func in [('latency_mean_ms', np.nanmean), ('latency_p50_ms', lambda x: np.nanpercentile(x, 50)),
                           ('latency_p95_ms', lambda x: np.nanpercentile(x, 95))]:
            entry[stat] = safe_stat(latency, func)

        # CPU %
        cpu = group['cpu_pct'] if 'cpu_pct' in group else pd.Series(dtype=float)
        for stat, func in [('cpu_mean_pct', np.nanmean), ('cpu_p95_pct', lambda x: np.nanpercentile(x, 95)), ('cpu_max_pct', np.nanmax)]:
            entry[stat] = safe_stat(cpu, func)

        # RAM MB
        ram = group['ram_mb'] if 'ram_mb' in group else pd.Series(dtype=float)
        for stat, func in [('ram_mean_mb', np.nanmean), ('ram_p95_mb', lambda x: np.nanpercentile(x, 95)), ('ram_max_mb', np.nanmax)]:
            entry[stat] = safe_stat(ram, func)

        # Detection rate (% of frames with detections > 0, one decimal)
        if 'detections' in group:
            num_detect = np.sum(group['detections'] > 0)
            rate = 100 * num_detect / group.shape[0] if group.shape[0] > 0 else np.nan
            entry['detection_rate'] = round(rate, 1)
        else:
            entry['detection_rate'] = np.nan

        result.append(entry)

    # Order methods by fps_mean descending
    result.sort(key=lambda x: (x.get('fps_mean', -np.inf) if pd.notna(x.get('fps_mean')) else -np.inf), reverse=True)
    return pd.DataFrame(result)

def safe_stat(series, func):
    if series.isnull().all():
        return np.nan
    try:
        val = func(series)
        if np.isnan(val):
            return np.nan
        return float(val)
    except Exception:
        return np.nan

def save_tables(df_agg, tables_dir):
    os.makedirs(tables_dir, exist_ok=True)
    # Clean up for export (empty cell if all NaN)
    df_export = df_agg.copy()
    for col in df_export.columns:
        df_export[col] = df_export[col].apply(lambda v: "" if pd.isna(v) else v)
    # Save CSV
    csv_path = os.path.join(tables_dir, "section4_summary.csv")
    df_export.to_csv(csv_path, index=False)
    # Save markdown
    md_path = os.path.join(tables_dir, "section4_summary.md")
    with open(md_path, "w") as f:
        f.write(df_export.to_markdown(index=False))

# --- Plotting Functions ---

def plot_box(df, metric, ylabel, out_dir, fmt, dpi):
    plt.figure(figsize=(8, 5))
    data = [df[df['Method'] == m][metric].dropna() for m in df['Method'].unique()]
    plt.boxplot(data, labels=df['Method'].unique(), patch_artist=True, showmeans=True)
    plt.ylabel(ylabel)
    plt.xlabel("Method")
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.title(f"{metric.replace('_', ' ').capitalize()} by Method")
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"box_{metric}.{'png' if fmt=='png' else fmt}")
    plt.savefig(out_path, bbox_inches='tight', dpi=dpi)
    plt.close()

def plot_hist(df, metric, out_dir, fmt, dpi, xlog=False):
    methods = df['Method'].unique()
    plt.figure(figsize=(7, 4))
    bins = 30
    for m in methods:
        vals = df[df['Method'] == m][metric].dropna()
        if len(vals) == 0:
            continue
        plt.hist(vals, bins=bins, alpha=0.5, label=m, histtype='stepfilled', linewidth=1.5)
    plt.xlabel(metric.replace('_', ' ').capitalize())
    plt.ylabel("Count")
    if xlog:
        plt.xscale("log")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.title(f"Histogram of {metric.replace('_', ' ')} (all methods)")
    plt.legend()
    plt.tight_layout()
    fname = f"hist_{metric}.{'png' if fmt=='png' else fmt}"
    plt.savefig(os.path.join(out_dir, fname), bbox_inches='tight', dpi=dpi)
    plt.close()

    # Individual method hists
    for m in methods:
        vals = df[df['Method'] == m][metric].dropna()
        if len(vals) == 0:
            continue
        plt.figure(figsize=(6, 4))
        plt.hist(vals, bins=bins, color='C0', alpha=0.8, edgecolor='k')
        plt.xlabel(metric.replace('_', ' ').capitalize())
        plt.ylabel("Count")
        if xlog:
            plt.xscale("log")
        plt.title(f"{m}: Histogram of {metric.replace('_', ' ')}")
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        fname = f"hist_{metric}_{m.replace(' ', '_')}.{'png' if fmt=='png' else fmt}"
        plt.savefig(os.path.join(out_dir, fname), bbox_inches='tight', dpi=dpi)
        plt.close()

def plot_summary_bars(df_agg, metric, ylabel, title, out_dir, fname, fmt, dpi, annotate=True, sort_desc=True):
    df_plot = df_agg[['Method', metric]].copy()
    # filter NaN
    df_plot = df_plot[df_plot[metric].notna()]
    if sort_desc:
        df_plot = df_plot.sort_values(metric, ascending=False)
    else:
        df_plot = df_plot.sort_values(metric, ascending=True)
    plt.figure(figsize=(8, 4))
    bars = plt.bar(df_plot['Method'], df_plot[metric], color='C0', alpha=0.7, edgecolor='k')
    plt.ylabel(ylabel)
    plt.xlabel("Method")
    plt.title(title)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    if annotate:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                plt.annotate(f"{height:.1f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, fname), bbox_inches='tight', dpi=dpi)
    plt.close()

def plot_grouped_bars(df_agg, metric1, metric2, ylabel, title, out_dir, fname, fmt, dpi):
    methods = df_agg['Method']
    vals1 = df_agg[metric1].values
    vals2 = df_agg[metric2].values
    x = np.arange(len(methods))
    width = 0.35
    plt.figure(figsize=(9, 4))
    bar1 = plt.bar(x - width/2, vals1, width, label=metric1.replace('_', ' '), color='C0', alpha=0.7, edgecolor='k')
    bar2 = plt.bar(x + width/2, vals2, width, label=metric2.replace('_', ' '), color='C1', alpha=0.7, edgecolor='k')
    plt.ylabel(ylabel)
    plt.xlabel("Method")
    plt.title(title)
    plt.xticks(x, methods, rotation=0)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    # annotate
    for bars in [bar1, bar2]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                plt.annotate(f"{height:.1f}", xy=(bar.get_x() + bar.get_width()/2, height),
                             xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, fname), bbox_inches='tight', dpi=dpi)
    plt.close()

# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Section 4: Generate summary tables and plots from metrics CSV")
    parser.add_argument("--input", type=str, default="Metrics/out/section4_metrics_all.csv", help="Input CSV file path")
    parser.add_argument("--out", type=str, default="Metrics/out", help="Output base directory for graphs")
    parser.add_argument("--tables", type=str, default="Metrics/out/tables", help="Output directory for tables")
    parser.add_argument("--format", type=str, choices=["png", "pdf"], default="png", help="Image format")
    parser.add_argument("--dpi", type=int, default=120, help="Image DPI")
    parser.add_argument("--only", type=str, nargs="*", default=None, help="Only include these methods")
    args = parser.parse_args()

    # Output dirs
    graphs_dir = os.path.join(args.out, "graphs")
    os.makedirs(graphs_dir, exist_ok=True)
    os.makedirs(args.tables, exist_ok=True)

    # Load data
    df = load_data(args.input, only_methods=args.only)
    methods = df['Method'].unique()

    # Compute aggregate summary table
    df_agg = compute_aggregates(df)

    # Save summary table as CSV and Markdown
    save_tables(df_agg, args.tables)

    # --- Plots ---

    # Boxplots
    plot_box(df, 'latency_ms', 'Inference Latency [ms]', graphs_dir, args.format, args.dpi)
    plot_box(df, 'fps_inst', 'Instantaneous FPS', graphs_dir, args.format, args.dpi)

    # Histograms (combined and per method)
    plot_hist(df, 'latency_ms', graphs_dir, args.format, args.dpi, xlog=True)
    plot_hist(df, 'fps_inst', graphs_dir, args.format, args.dpi, xlog=False)

    # Summary bar charts
    plot_summary_bars(
        df_agg, 'latency_p95_ms', 'P95 Latency [ms]',
        "P95 Inference Latency per Method", graphs_dir, "resumen_latency_ms."+args.format, args.format, args.dpi
    )
    plot_summary_bars(
        df_agg, 'fps_mean', 'Mean FPS',
        "Mean FPS per Method", graphs_dir, "resumen_fps_inst."+args.format, args.format, args.dpi
    )
    # Grouped bars for CPU
    plot_grouped_bars(
        df_agg, 'cpu_mean_pct', 'cpu_p95_pct', 'CPU [%]',
        "CPU Usage per Method (mean, P95)", graphs_dir, "resumen_cpu_pct."+args.format, args.format, args.dpi
    )
    # Grouped bars for RAM
    plot_grouped_bars(
        df_agg, 'ram_mean_mb', 'ram_p95_mb', 'RAM [MB]',
        "RAM Usage per Method (mean, P95)", graphs_dir, "resumen_ram_mb."+args.format, args.format, args.dpi
    )

    print("Plots and summary tables generated in", args.out, args.tables)

if __name__ == "__main__":
    main()