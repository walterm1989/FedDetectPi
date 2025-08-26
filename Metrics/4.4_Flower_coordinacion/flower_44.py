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
    Uses columns: fps_inst, latency_ms, cpu_pct, ram_mb, detection_flag.

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

    # coverage: percent of samples where detection_flag >= 1
    if 'detection_flag' in method_df.columns:
        coverage_pct = 100.0 * (method_df['detection_flag'] >= 1).sum() / len(method_df) if len(method_df) > 0 else np.nan
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

def make_note_coord():
    """Stub for coordination note generation."""
    pass

def save_tables(metrics, overhead, out_dir):
    """Stub for saving tables."""
    pass

def plotting(metrics, figs_dir):
    """Stub for plotting functions."""
    pass

def write_readme():
    """Stub for writing a README file."""
    pass

def main():
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
    args = parser.parse_args()

    # Ensure output directories exist
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.tables, exist_ok=True)

    # Stub main logic
    data = load_data(args)
    metrics = compute_metrics(data)
    overhead = compute_overhead(data)
    make_note_coord()
    save_tables(metrics, overhead, args.tables)
    plotting(metrics, args.out)
    if args.make_readme:
        write_readme()

if __name__ == "__main__":
    main()