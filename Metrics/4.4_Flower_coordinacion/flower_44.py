#!/usr/bin/env python3

import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data(path, only_methods=None):
    """
    Load and preprocess metrics data from a CSV file.

    Args:
        path (str): Path to the CSV file.
        only_methods (list or str, optional): List of method names (or comma-separated string) to filter by.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    # Read CSV
    df = pd.read_csv(path)

    # Strip and lowercase column names
    df.columns = [col.strip().lower() for col in df.columns]

    # Rename id_elapsed_sec or elapsed_sec to frame_id if present
    if 'id_elapsed_sec' in df.columns:
        df.rename(columns={'id_elapsed_sec': 'frame_id'}, inplace=True)
    elif 'elapsed_sec' in df.columns:
        df.rename(columns={'elapsed_sec': 'frame_id'}, inplace=True)

    # Convert numeric columns (except 'method', 'frame_id')
    non_numeric = {'method', 'frame_id'}
    for col in df.columns:
        if col not in non_numeric:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Filter by only_methods if provided
    if only_methods is not None:
        if isinstance(only_methods, str):
            only_list = [m.strip() for m in only_methods.split(',') if m.strip()]
        else:
            only_list = list(only_methods)
        df = df[df['method'].isin(only_list)]

    return df

def compute_metrics(df, method):
    """
    Compute summary metrics for a given method.
    Uses columns: fps_inst, latency_ms, cpu_pct, ram_mb, detection_flag or detections.

    Args:
        df (pd.DataFrame): Input dataframe.
        method (str): Method name to filter and compute metrics for.

    Returns:
        dict: Dictionary with required metrics.
            - fps_mean
            - latency_p95_ms
            - cpu_mean_pct
            - cpu_p95_pct
            - ram_mean_mb
            - ram_p95_mb
            - coverage_pct
    """
    import numpy as np

    # Filter dataframe by method
    method_df = df[df['method'] == method] if method in df['method'].unique() else df.copy()

    def safe_mean(s): return np.nanmean(s) if not s.empty else np.nan
    def safe_p95(s): return np.nanpercentile(s, 95) if not s.empty else np.nan

    fps_mean = safe_mean(method_df['fps_inst']) if 'fps_inst' in method_df.columns else np.nan
    latency_p95_ms = safe_p95(method_df['latency_ms']) if 'latency_ms' in method_df.columns else np.nan
    cpu_mean_pct = safe_mean(method_df['cpu_pct']) if 'cpu_pct' in method_df.columns else np.nan
    cpu_p95_pct = safe_p95(method_df['cpu_pct']) if 'cpu_pct' in method_df.columns else np.nan
    ram_mean_mb = safe_mean(method_df['ram_mb']) if 'ram_mb' in method_df.columns else np.nan
    ram_p95_mb = safe_p95(method_df['ram_mb']) if 'ram_mb' in method_df.columns else np.nan

    # coverage: percent of samples where detection_flag >= 1, or detections >= 1, else np.nan
    if 'detection_flag' in method_df.columns:
        coverage_pct = 100.0 * (method_df['detection_flag'] >= 1).mean() if len(method_df) > 0 else np.nan
    elif 'detections' in method_df.columns:
        coverage_pct = 100.0 * (method_df['detections'] >= 1).mean() if len(method_df) > 0 else np.nan
    else:
        coverage_pct = np.nan

    return {
        'fps_mean': fps_mean,
        'latency_p95_ms': latency_p95_ms,
        'cpu_mean_pct': cpu_mean_pct,
        'cpu_p95_pct': cpu_p95_pct,
        'ram_mean_mb': ram_mean_mb,
        'ram_p95_mb': ram_p95_mb,
        'coverage_pct': coverage_pct
    }

def compute_overhead(base_metrics, coord_metrics, cpu_ohw, ram_ohw, fps_drop):
    """
    Compute overhead metrics between base and coordination runs with updated keys.

    Args:
        base_metrics (dict): Baseline metrics.
        coord_metrics (dict): Coordination metrics.
        cpu_ohw (float): Allowed CPU overhead.
        ram_ohw (float): Allowed RAM overhead.
        fps_drop (float): Allowed FPS drop (percent, negative).

    Returns:
        dict: Deltas and result.
            - delta_cpu_mean
            - delta_ram_mean
            - delta_fps_mean
            - delta_coverage
            - cpu_ohw
            - ram_ohw
            - fps_drop
            - result ('OK' or 'Revisar ajustes')
    """
    import numpy as np

    try:
        cpu_base = base_metrics.get('cpu_mean_pct', np.nan)
        cpu_coord = coord_metrics.get('cpu_mean_pct', np.nan)
        ram_base = base_metrics.get('ram_mean_mb', np.nan)
        ram_coord = coord_metrics.get('ram_mean_mb', np.nan)
        fps_base = base_metrics.get('fps_mean', np.nan)
        fps_coord = coord_metrics.get('fps_mean', np.nan)
        cov_base = base_metrics.get('coverage_pct', np.nan)
        cov_coord = coord_metrics.get('coverage_pct', np.nan)
    except Exception:
        cpu_base = cpu_coord = ram_base = ram_coord = fps_base = fps_coord = cov_base = cov_coord = np.nan

    # Compute deltas (coord - base for cpu, ram, coverage; percent change for fps)
    delta_cpu_mean = cpu_coord - cpu_base if not (np.isnan(cpu_coord) or np.isnan(cpu_base)) else np.nan
    delta_ram_mean = ram_coord - ram_base if not (np.isnan(ram_coord) or np.isnan(ram_base)) else np.nan
    delta_fps_mean = ((fps_coord - fps_base) / fps_base * 100) if not (np.isnan(fps_coord) or np.isnan(fps_base) or fps_base == 0) else np.nan
    delta_coverage = cov_coord - cov_base if not (np.isnan(cov_coord) or np.isnan(cov_base)) else np.nan

    # Result: all deltas within allowed overheads (absolute for cpu/ram, >= for fps drop)
    if (
        not np.isnan(delta_cpu_mean) and not np.isnan(delta_ram_mean) and not np.isnan(delta_fps_mean)
        and abs(delta_cpu_mean) <= cpu_ohw
        and abs(delta_ram_mean) <= ram_ohw
        and delta_fps_mean >= -abs(fps_drop)
    ):
        result = 'OK'
    elif (
        np.isnan(delta_cpu_mean) or np.isnan(delta_ram_mean) or np.isnan(delta_fps_mean)
    ):
        result = 'Revisar ajustes'
    else:
        result = 'Revisar ajustes'

    return {
        'delta_cpu_mean': delta_cpu_mean,
        'delta_ram_mean': delta_ram_mean,
        'delta_fps_mean': delta_fps_mean,
        'delta_coverage': delta_coverage,
        'cpu_ohw': cpu_ohw,
        'ram_ohw': ram_ohw,
        'fps_drop': fps_drop,
        'result': result
    }

def make_note_coord(method, deltas, thresholds):
    """
    Generate a note summarizing the coordination deltas and thresholds.

    Args:
        method (str): Method name.
        deltas (dict): Dictionary of deltas and result.
        thresholds (dict): Dictionary of thresholds.

    Returns:
        str: Formatted note string.
    """
    msg = f"""### Coordinación: Método {method}

Resultados respecto al baseline:

- ΔCPU: {deltas['delta_cpu_mean']:.2f}% (límite {thresholds['cpu_ohw']}%)
- ΔRAM: {deltas['delta_ram_mean']:.2f} MB (límite {thresholds['ram_ohw']} MB)
- ΔFPS: {deltas['delta_fps_mean']:.2f}% (límite -{thresholds['fps_drop']}%)
- ΔCobertura: {deltas['delta_coverage']:.2f} pp

Resultado: **{deltas['result']}**

"""
    if deltas["result"] != "OK":
        msg += "\n> Revisar los ajustes de coordinación, se sobrepasaron uno o más límites.\n"
    else:
        msg += "\n> La coordinación cumple con los límites establecidos.\n"
    return msg

def save_tables(metrics_base, metrics_coord, deltas, tables_dir):
    """
    Save summary tables for baseline, coordination, and deltas.

    Args:
        metrics_base (dict): Baseline metrics.
        metrics_coord (dict): Coordination metrics.
        deltas (dict): Deltas dictionary.
        tables_dir (str): Output directory for tables.
    """
    import pandas as pd
    # Save baseline and coordination as single-row CSVs
    pd.DataFrame([metrics_base]).to_csv(os.path.join(tables_dir, "metrics_baseline.csv"), index=False)
    pd.DataFrame([metrics_coord]).to_csv(os.path.join(tables_dir, "metrics_coord.csv"), index=False)
    pd.DataFrame([deltas]).to_csv(os.path.join(tables_dir, "metrics_deltas.csv"), index=False)

def plot_bar_comparison(metrics_base, metrics_coord, out_dir, fmt, dpi):
    """
    Bar plot comparing metrics baseline vs coordinación.

    Args:
        metrics_base (dict), metrics_coord (dict), out_dir (str), fmt (str), dpi (int)
    """
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [
        "fps_mean", "latency_p95_ms",
        "cpu_mean_pct", "cpu_p95_pct",
        "ram_mean_mb", "ram_p95_mb", "coverage_pct"
    ]
    metric_names = [
        "FPS promedio", "Latencia 95° (ms)",
        "CPU promedio (%)", "CPU 95° (%)",
        "RAM promedio (MB)", "RAM 95° (MB)", "Cobertura (%)"
    ]
    base_vals = [metrics_base[k] for k in labels]
    coord_vals = [metrics_coord[k] for k in labels]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(10,6))
    plt.bar(x-width/2, base_vals, width, label='Baseline')
    plt.bar(x+width/2, coord_vals, width, label='Coordinación')
    plt.xticks(x, metric_names, rotation=15)
    plt.ylabel("Valor")
    plt.title("Comparación de métricas: Baseline vs Coordinación")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"bar_comparison.{fmt}"), dpi=dpi)
    plt.close()

def plot_threshold_vs_metric(df, method, threshold_col, metric_col, out_dir, fmt, dpi):
    """
    Plot threshold vs metric (e.g., FPS or Cobertura) for the specified method.

    Args:
        df (pd.DataFrame), method (str), threshold_col (str), metric_col (str), out_dir (str), fmt (str), dpi (int)
    """
    import matplotlib.pyplot as plt

    data = df[df['method'] == method]
    if threshold_col not in data.columns or metric_col not in data.columns:
        return

    plt.figure(figsize=(7,5))
    plt.plot(data[threshold_col], data[metric_col], marker='o')
    plt.xlabel(threshold_col)
    plt.ylabel(metric_col)
    plt.title(f"{metric_col} vs {threshold_col} ({method})")
    plt.grid(True, ls='--', alpha=0.6)
    plt.tight_layout()
    fname = f"thresh_vs_{metric_col}.{fmt}"
    plt.savefig(os.path.join(out_dir, fname), dpi=dpi)
    plt.close()

import re

def plot_timeline(df, method, out_dir, fmt, dpi, no_smooth=False, suffix=''):
    """
    Plot FPS and detection timeline for the specified method.

    Args:
        df (pd.DataFrame), method (str), out_dir (str), fmt (str), dpi (int), no_smooth (bool), suffix (str)
    """
    import matplotlib.pyplot as plt

    # Make slug for method
    slug = re.sub(r'[^A-Za-z0-9]+', '_', method).strip('_')
    # Ensure timelines out_dir exists
    os.makedirs(out_dir, exist_ok=True)
    # Filename with optional suffix
    fname = f"timeline_{slug}{('_'+suffix) if suffix else ''}.{fmt}"

    data = df[df['method'] == method]
    if 'frame_id' not in data.columns:
        return

    fig, ax1 = plt.subplots(figsize=(12, 5))

    # Always plot fps_inst on ax1
    if 'fps_inst' in data.columns:
        ax1.plot(data['frame_id'], data['fps_inst'], 'b-', label="FPS inst")
        ax1.set_ylabel('FPS inst', color='b')
        ax1.tick_params(axis='y', labelcolor='b')

    ax2 = None
    # Only plot detection smoothed series if no_smooth is False and detection data exists
    if not no_smooth:
        series = None
        if 'detection_flag' in data.columns:
            series = data['detection_flag']
        elif 'detections' in data.columns:
            series = (data['detections'] > 0).astype(float)
        # Only proceed if we have detection data
        if series is not None:
            smoothed = series.rolling(window=5, min_periods=1).mean()
            ax2 = ax1.twinx()
            ax2.plot(data['frame_id'], smoothed, color='orange', label="Detección (suavizado)")
            ax2.set_ylabel("Detección (0–1)", color='orange')
            ax2.set_ylim([0, 1])
            ax2.tick_params(axis='y', labelcolor='orange')

    # Add legends
    ax1.legend(loc='upper left')
    if ax2:
        ax2.legend(loc='upper right')

    # Enable grid
    ax1.grid(True, alpha=0.25)

    # Set title
    plt.title(f"Timeline FPS y Detección ({method})")

    # Tight layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, fname), dpi=dpi, bbox_inches='tight')
    plt.close()

def write_readme(args, results):
    """
    Write a README.md summarizing inputs, results and deltas.

    Args:
        args: argparse arguments.
        results: dict with method, metrics, deltas, result, paths.
    """
    readme_path = os.path.join(args.out, "README.md")
    mode = results.get('mode', 'paired' if getattr(args, "coord_input", None) else 'simple')
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(f"# Coordinación FlowerAI 4.4\n")
        f.write("\n")
        f.write(f"**Modo:** {mode}\n")
        f.write("\n")
        f.write(f"**Método:** `{results['method']}`\n")
        f.write("\n")
        f.write(f"**Archivo baseline:** `{args.baseline_input}`\n")
        f.write("\n")
        if mode == "paired" and getattr(args, "coord_input", None):
            f.write(f"**Archivo coordinación:** `{args.coord_input}`\n\n")
            f.write("## Métricas Baseline\n")
            for k, v in results['baseline'].items():
                f.write(f"- {k}: {v:.3f}\n")
            f.write("\n## Métricas Coordinación\n")
            for k, v in results['coord'].items():
                f.write(f"- {k}: {v:.3f}\n")
            f.write("\n## Deltas\n")
            for k in ['delta_cpu_mean', 'delta_ram_mean', 'delta_fps_mean', 'delta_coverage']:
                f.write(f"- {k}: {results['deltas'][k]:.3f}\n")
            f.write(f"\n## Resultado: **{results['deltas']['result']}**\n\n")
            f.write(make_note_coord(results['method'], results['deltas'], {
                'cpu_ohw': args.cpu_ohw,
                'ram_ohw': args.ram_ohw,
                'fps_drop': args.fps_drop
            }))
            f.write("\n---\n")
            f.write("Figuras y tablas generadas en las carpetas correspondientes.\n")
        else:
            # simple mode
            f.write("## Métricas Baseline\n")
            for k, v in results['baseline'].items():
                f.write(f"- {k}: {v:.3f}\n")
            f.write("\nNota: Para analizar overhead, ejecutar en modo emparejado con --coord-input.\n")

def save_simple_baseline_table_markdown(metrics, tables_dir):
    """
    Save tabla_44_coordinacion.csv and .md for simple mode.

    Args:
        metrics (dict): Baseline metrics.
        tables_dir (str): Output directory for tables.
    """

    # Save CSV
    csv_path = os.path.join(tables_dir, "tabla_44_coordinacion.csv")
    pd.DataFrame([metrics], index=["baseline"]).to_csv(csv_path, index_label="key")
    
    # Save Markdown
    md_path = os.path.join(tables_dir, "tabla_44_coordinacion.md")
    with open(md_path, "w", encoding="utf-8") as f:
        headers = ["key"] + list(metrics.keys())
        values = ["baseline"] + [f"{v:.3f}" if isinstance(v, float) else str(v) for v in metrics.values()]
        # Markdown table header
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + " --- |" * len(headers) + "\n")
        # Markdown table row
        f.write("| " + " | ".join(values) + " |\n")

def main():
    import warnings

    parser = argparse.ArgumentParser(description="Flower 4.4 Coordination Metrics")
    parser.add_argument(
        "--baseline-input",
        type=str,
        default="Metrics/out/section4_metrics_all.csv",
        help="Path to baseline input CSV (default: Metrics/out/section4_metrics_all.csv)"
    )
    parser.add_argument(
        "--coord-input",
        type=str,
        default=None,
        help="Path to coordination input CSV (optional)"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="BBoxes-YOLOv4tiny",
        help="Method to use (default: BBoxes-YOLOv4tiny)"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="Metrics/4.4_Flower_coordinacion/figs",
        help="Directory to save output figures (default: Metrics/4.4_Flower_coordinacion/figs)"
    )
    parser.add_argument(
        "--tables",
        type=str,
        default="Metrics/4.4_Flower_coordinacion/tables",
        help="Directory to save output tables (default: Metrics/4.4_Flower_coordinacion/tables)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["png", "pdf"],
        default="png",
        help="Format for output figures (png or pdf, default: png)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI for output figures (default: 200)"
    )
    parser.add_argument(
        "--no-smooth",
        action="store_true",
        help="If set, do not plot the smoothed detection series and omit secondary axis"
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Optional: Only process the specified section"
    )
    parser.add_argument(
        "--cpu-ohw",
        type=float,
        default=5.0,
        help="CPU overhead value (default: 5.0)"
    )
    parser.add_argument(
        "--ram-ohw",
        type=int,
        default=50,
        help="RAM overhead value (default: 50)"
    )
    parser.add_argument(
        "--fps-drop",
        type=float,
        default=10.0,
        help="FPS drop threshold (default: 10.0)"
    )
    parser.add_argument(
        "--threshold-column",
        type=str,
        default="threshold",
        help="Column name for threshold (default: threshold)"
    )
    parser.add_argument(
        "--make-readme",
        action="store_true",
        help="If set, generate a README file"
    )
    # --- new batch-all mode ---
    parser.add_argument(
        "--batch-all",
        action="store_true",
        help="If set, batch process all methods in standardized directory"
    )
    parser.add_argument(
        "--std-dir",
        type=str,
        default="Metrics/out/standardized",
        help="Directory for standardized CSVs (default: Metrics/out/standardized)"
    )
    args = parser.parse_args()

    # Ensure output directories exist
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.tables, exist_ok=True)

    # --------- BATCH-ALL MODE ---------
    if args.batch_all:
        import glob

        print(f"[INFO] Batch-all mode enabled.")
        std_dir = args.std_dir
        # New: Centralized method to prefix mapping
        method_prefix_map = {
            'BBoxes-YOLOv4tiny': 'bboxes',
            'KeyPoints-ResNet50': 'keypoints',
            'FlowerAI': 'flower'
        }
        methods = list(method_prefix_map.keys())

        results_dict = {}
        comparative_rows = []
        columns = [
            "Esquema",
            "FPS base", "FPS coord", "ΔFPS (%)",
            "CPU base (%)", "CPU coord (%)", "ΔCPU (%)",
            "RAM base (MB)", "RAM coord (MB)", "ΔRAM (MB)",
            "Cobertura base (%)", "Cobertura coord (%)", "ΔCobertura (p.p.)",
            "Aplicabilidad práctica (y mejoras)"
        ]

        deltas_data = { "method": [], "delta_fps": [], "delta_cpu": [], "delta_ram": [], "delta_cov": [] }
        markdown_lines = []
        markdown_lines.append("| " + " | ".join(columns) + " |")
        markdown_lines.append("|" + " --- |" * len(columns))

        timelines_dir = os.path.join(args.out, "timelines")
        os.makedirs(timelines_dir, exist_ok=True)

        thresholds = {
            'cpu_ohw': args.cpu_ohw,
            'ram_ohw': args.ram_ohw,
            'fps_drop': args.fps_drop
        }

        for method, prefix in method_prefix_map.items():
            base_file = os.path.join(std_dir, f"{prefix}_std.csv")
            coord_file = os.path.join(std_dir, f"{prefix}_std_coord.csv")

            if not os.path.exists(base_file):
                warnings.warn(f"Missing baseline file {base_file} for method {method}. Skipping.")
                continue
            if not os.path.exists(coord_file):
                warnings.warn(f"Missing coordination file {coord_file} for method {method}. Skipping.")
                continue

            try:
                df_base = load_data(base_file, only_methods=[method])
                df_coord = load_data(coord_file, only_methods=[method])
            except Exception as e:
                warnings.warn(f"Error loading files for {method}: {e}")
                continue

            metrics_base = compute_metrics(df_base, method)
            metrics_coord = compute_metrics(df_coord, method)
            deltas = compute_overhead(metrics_base, metrics_coord, args.cpu_ohw, args.ram_ohw, args.fps_drop)
            note = make_note_coord(method, deltas, thresholds).replace("\n", " ").replace("|", "/")

            row = [
                method,
                f"{metrics_base['fps_mean']:.2f}", f"{metrics_coord['fps_mean']:.2f}", f"{deltas['delta_fps_mean']:.2f}",
                f"{metrics_base['cpu_mean_pct']:.2f}", f"{metrics_coord['cpu_mean_pct']:.2f}", f"{deltas['delta_cpu_mean']:.2f}",
                f"{metrics_base['ram_mean_mb']:.2f}", f"{metrics_coord['ram_mean_mb']:.2f}", f"{deltas['delta_ram_mean']:.2f}",
                f"{metrics_base['coverage_pct']:.2f}", f"{metrics_coord['coverage_pct']:.2f}", f"{deltas['delta_coverage']:.2f}",
                note.strip()
            ]
            comparative_rows.append(row)
            markdown_lines.append("| " + " | ".join(row) + " |")
            deltas_data["method"].append(method)
            deltas_data["delta_fps"].append(float(deltas['delta_fps_mean']))
            deltas_data["delta_cpu"].append(float(deltas['delta_cpu_mean']))
            deltas_data["delta_ram"].append(float(deltas['delta_ram_mean']))
            deltas_data["delta_cov"].append(float(deltas['delta_coverage']))
            results_dict[method] = {
                "metrics_base": metrics_base,
                "metrics_coord": metrics_coord,
                "deltas": deltas,
            }

            # Timeline plots for each method (use suffixes)
            try:
                plot_timeline(df_base, method, timelines_dir, args.format, args.dpi, args.no_smooth, suffix='baseline')
            except Exception as e:
                warnings.warn(f"Could not plot baseline timeline for {method}: {e}")
            try:
                plot_timeline(df_coord, method, timelines_dir, args.format, args.dpi, args.no_smooth, suffix='coord')
            except Exception as e:
                warnings.warn(f"Could not plot coordination timeline for {method}: {e}")

        # === Save comparative CSV and Markdown ===
        tabla_csv = os.path.join(args.tables, "tabla_44_coordinacion_comparativa.csv")
        tabla_md  = os.path.join(args.tables, "tabla_44_coordinacion_comparativa.md")
        import csv
        # CSV
        with open(tabla_csv, "w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            for row in comparative_rows:
                writer.writerow(row)
        # Markdown
        with open(tabla_md, "w", encoding="utf-8") as f:
            for line in markdown_lines:
                f.write(line + "\n")

        # === Delta bar charts ===
        import matplotlib.pyplot as plt
        import numpy as np

        def plot_delta_bar(key, ylabel, fname):
            plt.figure(figsize=(8,5))
            vals = deltas_data[key]
            x = np.arange(len(deltas_data["method"]))
            plt.bar(x, vals, color="skyblue")
            plt.xticks(x, deltas_data["method"], rotation=15)
            plt.ylabel(ylabel)
            plt.title(f"Δ{ylabel} (Coordinación - Baseline)")
            plt.tight_layout()
            plt.savefig(os.path.join(args.out, fname), dpi=args.dpi, bbox_inches='tight')
            plt.close()

        plot_delta_bar("delta_fps", "FPS (%)", "fig_44_deltas_fps." + args.format)
        plot_delta_bar("delta_cpu", "CPU (%)", "fig_44_deltas_cpu." + args.format)
        plot_delta_bar("delta_ram", "RAM (MB)", "fig_44_deltas_ram." + args.format)
        plot_delta_bar("delta_cov", "Cobertura (p.p.)", "fig_44_deltas_cobertura." + args.format)

        # === README_4.4.md ===
        readme_path = os.path.join(os.path.dirname(__file__), "README_4.4.md")
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(f"# Coordinación FlowerAI 4.4 - Batch All\n\n")
            f.write("Este informe resume la comparación de métricas entre el baseline y la coordinación para todos los esquemas (métodos) analizados.\n\n")
            f.write("## Umbrales utilizados\n")
            f.write(f"- CPU overhead permitido: {args.cpu_ohw:.2f}%\n")
            f.write(f"- RAM overhead permitido: {args.ram_ohw:.2f} MB\n")
            f.write(f"- Caída de FPS permitida: -{args.fps_drop:.2f}%\n\n")
            f.write("## Tablas comparativas\n\n")
            rel_tabla_csv = os.path.relpath(tabla_csv, os.path.dirname(readme_path))
            rel_tabla_md = os.path.relpath(tabla_md, os.path.dirname(readme_path))
            f.write(f"- [Tabla comparativa CSV]({rel_tabla_csv})\n")
            f.write(f"- [Tabla comparativa Markdown]({rel_tabla_md})\n\n")
            f.write("## Figuras de deltas\n\n")
            for suffix, desc in [
                ("fig_44_deltas_fps", "ΔFPS"),
                ("fig_44_deltas_cpu", "ΔCPU"),
                ("fig_44_deltas_ram", "ΔRAM"),
                ("fig_44_deltas_cobertura", "ΔCobertura"),
            ]:
                fname = suffix + "." + args.format
                rel_path = os.path.relpath(os.path.join(args.out, fname), os.path.dirname(readme_path))
                f.write(f"- ![{desc}]({rel_path})\n")
            f.write("\n## Timelines por método\n\n")
            for method in deltas_data["method"]:
                timeline_path = os.path.relpath(os.path.join(timelines_dir, f"timeline.{args.format}"), os.path.dirname(readme_path))
                f.write(f"- {method}: `{os.path.join('figs/timelines', f'timeline.{args.format}')}`\n")
            f.write("\n---\n")
            f.write("Este reporte fue generado automáticamente.\n")

        print(f"[INFO] Batch-all mode completed. Comparative table at {tabla_csv}, Markdown at {tabla_md}")
        print(f"[INFO] Figures saved in {args.out}. README_4.4.md written.")
        return

    # --------- END BATCH-ALL ---------

    print(f"[INFO] Baseline input: {args.baseline_input}")
    if args.coord_input:
        print(f"[INFO] Coordination input: {args.coord_input}")
    print(f"[INFO] Method: {args.method}")
    print(f"[INFO] Output figures: {args.out}")
    print(f"[INFO] Output tables: {args.tables}")

    # Mode selection: paired (coord-input provided) or simple
    if args.coord_input:
        # Paired mode: compare baseline and coordination
        df_base = load_data(args.baseline_input, only_methods=args.only)
        df_coord = load_data(args.coord_input, only_methods=args.only)
        metrics_base = compute_metrics(df_base, args.method)
        metrics_coord = compute_metrics(df_coord, args.method)
        deltas = compute_overhead(metrics_base, metrics_coord, args.cpu_ohw, args.ram_ohw, args.fps_drop)
        print(f"[RESULT] Deltas: {deltas}")
        save_tables(metrics_base, metrics_coord, deltas, args.tables)
        plot_bar_comparison(metrics_base, metrics_coord, args.out, args.format, args.dpi)
        # Optional: plot threshold vs metrics if threshold column exists
        if args.threshold_column in df_base.columns and args.threshold_column in df_coord.columns:
            plot_threshold_vs_metric(df_base, args.method, args.threshold_column, "fps_inst", args.out, args.format, args.dpi)
            plot_threshold_vs_metric(df_coord, args.method, args.threshold_column, "coverage_pct", args.out, args.format, args.dpi)
        # Timeline
        plot_timeline(df_coord, args.method, args.out, args.format, args.dpi, args.no_smooth)
        results = {
            'mode': 'paired',
            'method': args.method,
            'baseline': metrics_base,
            'coord': metrics_coord,
            'deltas': deltas
        }
        if args.make_readme:
            write_readme(args, results)
    else:
        # Simple mode: only baseline metrics and timeline plot
        df = load_data(args.baseline_input, only_methods=args.only)
        metrics = compute_metrics(df, args.method)
        print(f"[RESULT] Métricas baseline: {metrics}")
        # Save only tabla_44_coordinacion.csv and .md, no other tables
        save_simple_baseline_table_markdown(metrics, args.tables)
        # Only plot timeline (no bar or threshold plots)
        plot_timeline(df, args.method, args.out, args.format, args.dpi, args.no_smooth)
        results = {
            'mode': 'simple',
            'method': args.method,
            'baseline': metrics
        }
        if args.make_readme:
            write_readme(args, results)

if __name__ == "__main__":
    main()