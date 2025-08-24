import argparse
import os
import sys
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict
from matplotlib.ticker import MaxNLocator

def normalize_columns(df):
    # Renombres para FlowerAI → columnas estándar del analizador
    rename_map = {
        'timestamp_iso': 'timestamp',
        'elapsed_s': 'elapsed_sec',
        'cpu_percent': 'cpu_pct',
        'rss_mb': 'ram_mb',
        'cam_fps': 'fps_inst',   # usamos cam_fps como fps_inst
        'cam_det': 'detections', # opcional, por consistencia
    }
    to_rename = {k: v for k, v in rename_map.items() if k in df.columns}
    if to_rename:
        df = df.rename(columns=to_rename)

    # Si falta fps_inst pero hay latency_ms, lo calculamos
    if ('fps_inst' not in df.columns or df['fps_inst'].isnull().all()) and 'latency_ms' in df.columns and not df['latency_ms'].isnull().all():
        df['fps_inst'] = 1000.0 / df['latency_ms']

    # Si falta latency_ms pero hay fps_inst, lo calculamos
    if ('latency_ms' not in df.columns or df['latency_ms'].isnull().all()) and 'fps_inst' in df.columns and not df['fps_inst'].isnull().all():
        safe_fps = df['fps_inst'].replace([0, np.inf, -np.inf], np.nan)
        df['latency_ms'] = 1000.0 / safe_fps

    # Asegura elapsed_sec si hay timestamp
    if ('elapsed_sec' not in df.columns or df['elapsed_sec'].isnull().all()) and 'timestamp' in df.columns:
        times = pd.to_datetime(df['timestamp'], errors='coerce')
        if times.notna().any():
            df['elapsed_sec'] = (times - times.min()).dt.total_seconds()

    return df

def infer_method_from_filename(f):
    base = os.path.basename(f).lower()
    if 'keypoint' in base:
        return 'KeyPoints'
    elif 'bbox' in base:
        return 'BBoxes'
    elif 'flower' in base:
        return 'FlowerAI'
    else:
        return os.path.splitext(os.path.basename(f))[0]

# METRICS = [
#     'elapsed_sec', 'fps_inst', 'latency_ms', 'cpu_pct', 'ram_mb', 'detections'
# ]
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import warnings

METRICS = ['latency_ms', 'fps_inst', 'cpu_pct', 'ram_mb']
SUMMARY_STATS = ['mean', 'median', 'p95', 'std', 'count']

def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def parse_args():
    parser = argparse.ArgumentParser(description="Plot and summarize metrics")
    parser.add_argument('--raw', type=str, required=True, help='Input directory with raw CSVs')
    parser.add_argument('--out', type=str, required=True, help='Base output directory')
    parser.add_argument('--x-mode', type=str, default='frame', choices=['frame', 'elapsed', 'timestamp'],
                        help="X axis for time series: frame (default), elapsed, timestamp")
    parser.add_argument('--methods', type=str, default=None,
                        help="Comma-separated list of methods to include (optional)")
    parser.add_argument('--sources', type=str, default=None,
                        help="Comma-separated list of sources to include (optional)")
    return parser.parse_args()

def get_csv_files(raw_path):
    patterns = [
        os.path.join(raw_path, '*.csv'),
        os.path.join(raw_path, 'metrics*.csv')
    ]
    files = set()
    for pattern in patterns:
        files.update(glob.glob(pattern))
    return sorted(files)

def normalize_columns(df):
    """
    Normalize DataFrame column headers and derive missing metrics if possible.
    This is especially useful for FlowerAI CSVs which may have non-standard headers.
    """
    # Example normalization: lowercase, strip, replace spaces with underscores
    df.columns = [c.lower().strip().replace(' ', '_') for c in df.columns]
    # FlowerAI-specific remapping
    mapping = {
        'total_accuracy': 'accuracy',
        'acc': 'accuracy',
        'elapsed': 'elapsed_sec',
        'duration': 'elapsed_sec',
        'cpu': 'cpu_pct',
        'ram': 'ram_pct',
        # Add more mappings as needed
    }
    df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})
    # Derive accuracy if not present but can be computed
    if 'correct' in df.columns and 'total' in df.columns and 'accuracy' not in df.columns:
        with pd.option_context('mode.use_inf_as_na', True):
            df['accuracy'] = df['correct'] / df['total']
    return df

def infer_method_from_filename(path):
    """
    Infer the 'method' name from a filename or path.
    Typical use: if filename is 'results_gpt4.csv', returns 'gpt4'
    """
    fname = os.path.basename(path)
    for delim in ['_', '-', '.']:
        parts = fname.split(delim)
        for part in parts:
            if part.lower() in ['gpt3', 'gpt4', 'flowerai', 'openai', 'llama', 'mistral', 'gemini']:
                return part
    # fallback: strip extension and return last part
    return fname.rsplit('.', 1)[0]

# --- User-specified functions (inserted verbatim) ---
import numpy as np
import pandas as pd

def normalize_columns(df):
    # Renombres para FlowerAI → columnas estándar del analizador
    rename_map = {
        'timestamp_iso': 'timestamp',
        'elapsed_s': 'elapsed_sec',
        'cpu_percent': 'cpu_pct',
        'rss_mb': 'ram_mb',
        'cam_fps': 'fps_inst',   # usamos cam_fps como fps_inst
        'cam_det': 'detections', # opcional, por consistencia
    }
    to_rename = {k: v for k, v in rename_map.items() if k in df.columns}
    if to_rename:
        df = df.rename(columns=to_rename)

    # Si falta fps_inst pero hay latency_ms, lo calculamos
    if 'fps_inst' not in df.columns or df['fps_inst'].isnull().all():
        if 'latency_ms' in df.columns and not df['latency_ms'].isnull().all():
            df['fps_inst'] = 1000.0 / df['latency_ms']

    # Si falta latency_ms pero hay fps_inst, lo calculamos
    if 'latency_ms' not in df.columns or df['latency_ms'].isnull().all():
        if 'fps_inst' in df.columns and not df['fps_inst'].isnull().all():
            safe_fps = df['fps_inst'].replace([0, np.inf, -np.inf], np.nan)
            df['latency_ms'] = 1000.0 / safe_fps

    # Asegura elapsed_sec si hay timestamp
    if 'elapsed_sec' not in df.columns or df['elapsed_sec'].isnull().all():
        if 'timestamp' in df.columns:
            times = pd.to_datetime(df['timestamp'], errors='coerce')
            if times.notna().any():
                df['elapsed_sec'] = (times - times.min()).dt.total_seconds()

    return df

def infer_method_from_filename(f):
    base = os.path.basename(f).lower()
    if 'keypoint' in base:
        return 'KeyPoints'
    elif 'bbox' in base:
        return 'BBoxes'
    elif 'flower' in base:
        return 'FlowerAI'
    else:
        return os.path.splitext(os.path.basename(f))[0]
# --- End user-specified functions ---

def fill_missing_columns(df):
    # Fill missing columns with NaN for compatibility
    cols = ['accuracy', 'throughput', 'latency', 'elapsed_sec', 'cpu_pct', 'ram_pct']
    for col in cols:
        if col not in df.columns:
            df[col] = float('nan')
    return df

def compute_elapsed_sec(df):
    # Create elapsed_sec = timestamp - first_timestamp, per file
    if 'elapsed_sec' not in df.columns or df['elapsed_sec'].isnull().all():
        if 'timestamp' in df.columns:
            try:
                times = pd.to_datetime(df['timestamp'], errors='coerce')
                first = times.min()
                df['elapsed_sec'] = (times - first).dt.total_seconds()
            except Exception as e:
                df['elapsed_sec'] = np.nan
        else:
            df['elapsed_sec'] = np.nan
    return df

def check_cpu_pct(df, num_cores=0):
    # If cpu_pct out of [0, 100*num_cores] or [0, 100] (if cores unknown), warn but don't truncate
    if 'cpu_pct' in df.columns:
        # Try to infer num_cores if not given
        if num_cores <= 0:
            # Guess based on observed max
            observed_max = df['cpu_pct'].max(skipna=True)
            if observed_max > 100:
                num_cores = int(np.ceil(observed_max / 100))
            else:
                num_cores = 1
        high = 100 * num_cores
        mask = (df['cpu_pct'] < 0) | (df['cpu_pct'] > high)
        if mask.any():
            warnings.warn(f"Some cpu_pct values are outside expected range [0, {high}]: {df['cpu_pct'][mask].tolist()}")
    return df

def load_all_data(raw_path, methods_filter=None, sources_filter=None):
    files = get_csv_files(raw_path)
    dfs = []
    file_info = []
    for f in files:
        try:
            df = pd.read_csv(f)
            # Normaliza encabezados y métricas (especialmente para FlowerAI)
            df = normalize_columns(df)

            # Método y fuente (limpios)
            if 'method' not in df.columns:
                df['method'] = infer_method_from_filename(f)
            if 'source' not in df.columns:
                df['source'] = os.path.basename(f)

            df = fill_missing_columns(df)
            df = compute_elapsed_sec(df)
            df = check_cpu_pct(df)
            # Save file info for summary
            run_start = pd.to_datetime(df['timestamp'], errors='coerce').min() if 'timestamp' in df.columns else pd.NaT
            run_duration = df['elapsed_sec'].max(skipna=True) if 'elapsed_sec' in df.columns else np.nan
            file_info.append({'method': df['method'].iloc[0], 'source': df['source'].iloc[0], 'run_start': run_start, 'run_duration_sec': run_duration})
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
    if not dfs:
        raise RuntimeError("No valid CSV files found in input directory.")
    big_df = pd.concat(dfs, ignore_index=True)
    # Optional filtering
    if methods_filter is not None:
        big_df = big_df[big_df['method'].isin(methods_filter)]
    if sources_filter is not None:
        big_df = big_df[big_df['source'].isin(sources_filter)]
    if big_df.empty:
        raise RuntimeError("No rows remain after filtering by methods/sources.")
    return big_df, pd.DataFrame(file_info)

def plot_time_series(df, metric, graphs_dir, x_mode='frame'):
    plt.figure(figsize=(10, 6))
    legend_methods = []
    for method, group in df.groupby('method'):
        group = group.reset_index(drop=True)
        if x_mode == 'frame':
            x = group['frame_idx'] if 'frame_idx' in group.columns else np.arange(len(group))
            xlabel = 'Frame Index'
        elif x_mode == 'elapsed':
            x = group['elapsed_sec'] if 'elapsed_sec' in group.columns else np.arange(len(group))
            xlabel = 'Elapsed Time (sec)'
        elif x_mode == 'timestamp':
            if 'timestamp' in group.columns:
                x = pd.to_datetime(group['timestamp'], errors='coerce')
                xlabel = 'Timestamp'
            else:
                x = np.arange(len(group))
                xlabel = 'Sample'
        else:
            x = np.arange(len(group))
            xlabel = 'Sample'
        plt.plot(x, group[metric], label=method)
        legend_methods.append(method)
    plt.xlabel(xlabel)
    plt.ylabel(metric.replace('_', ' ').capitalize())
    plt.title(f"{metric.replace('_',' ').capitalize()} vs {xlabel}")
    plt.legend(title='Method')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    fname = f"{metric}_vs_{x_mode}.png"
    out_path = os.path.join(graphs_dir, fname)
    plt.savefig(out_path)
    plt.close()

def plot_time_series_vs_time(df, metric, graphs_dir):
    # For compatibility: always produce *_vs_time.png as before
    plt.figure(figsize=(10, 6))
    for method, group in df.groupby('method'):
        if 'timestamp' in group.columns:
            group = group.sort_values('timestamp')
            x = pd.to_datetime(group['timestamp'], errors='coerce')
        else:
            x = range(len(group))
        plt.plot(x, group[metric], label=method)
    plt.xlabel('Timestamp' if 'timestamp' in df.columns else 'Sample')
    plt.ylabel(metric.replace('_', ' ').capitalize())
    plt.title(f"{metric.replace('_',' ').capitalize()} vs Time")
    plt.legend(title="Method")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    out_path = os.path.join(graphs_dir, f"{metric}_vs_time.png")
    plt.savefig(out_path)
    plt.close()

def plot_bar_summary(df, metric, graphs_dir):
    summary = df.groupby('method')[metric].agg(['mean', 'std'])
    methods = summary.index.tolist()
    means = summary['mean'].values
    stds = summary['std'].values

    plt.figure(figsize=(8, 6))
    bars = plt.bar(methods, means, yerr=stds, capsize=5, alpha=0.8, color='skyblue')
    plt.ylabel(metric.replace('_', ' ').capitalize())
    plt.title(f"Mean ± Std of {metric.replace('_', ' ').capitalize()} per Method")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    out_path = os.path.join(graphs_dir, f"resumen_{metric}.png")
    plt.savefig(out_path)
    plt.close()

def freedman_diaconis_bins(data):
    data = np.asarray(data)
    data = data[~np.isnan(data)]
    if len(data) < 2:
        return 10
    iqr = np.subtract(*np.percentile(data, [75, 25]))
    if iqr == 0:
        return 10
    bin_width = 2 * iqr / (len(data) ** (1/3))
    if bin_width == 0:
        return 10
    bins = int(np.ceil((data.max() - data.min()) / bin_width))
    return max(1, min(50, bins))

def plot_histogram(df, metric, graphs_dir):
    # Use consistent bins across methods
    all_vals = df[metric].dropna().values
    bins = min(20, freedman_diaconis_bins(all_vals))
    plt.figure(figsize=(8, 6))
    for method, group in df.groupby('method'):
        vals = group[metric].dropna().values
        plt.hist(vals, bins=bins, alpha=0.6, label=method, histtype='stepfilled', edgecolor='black')
    plt.xlabel(metric.replace('_', ' ').capitalize())
    plt.ylabel('Count')
    plt.title(f"Histogram of {metric.replace('_',' ').capitalize()} (bins={bins})")
    plt.legend(title='Method')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    out_path = os.path.join(graphs_dir, f"hist_{metric}.png")
    plt.savefig(out_path)
    plt.close()

def plot_boxplot(df, metric, graphs_dir):
    plt.figure(figsize=(8, 6))
    data = []
    labels = []
    for method, group in df.groupby('method'):
        vals = group[metric].dropna().values
        data.append(vals)
        labels.append(method)
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.ylabel(metric.replace('_', ' ').capitalize())
    plt.title(f"Boxplot of {metric.replace('_',' ').capitalize()} per Method")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    out_path = os.path.join(graphs_dir, f"box_{metric}.png")
    plt.savefig(out_path)
    plt.close()

def compute_summary_stats(df, file_info):
    def p95(x):
        return np.nanpercentile(x, 95)
    stat_funcs = {
        'mean': np.nanmean,
        'median': np.nanmedian,
        'p95': p95,
        'std': np.nanstd,
        'count': lambda x: np.count_nonzero(~np.isnan(x))
    }
    records = []
    for method, group in df.groupby('method'):
        rec = {'method': method}
        for metric in METRICS:
            vals = group[metric].values
            for stat, func in stat_funcs.items():
                rec[f'{metric}_{stat}'] = func(vals)
        # Find run_start and run_duration_sec per method from file_info
        starts = file_info[file_info['method'] == method]['run_start']
        durations = file_info[file_info['method'] == method]['run_duration_sec']
        rec['run_start'] = starts.min() if not starts.empty else pd.NaT
        rec['run_duration_sec'] = durations.max() if not durations.empty else np.nan
        records.append(rec)
    return pd.DataFrame.from_records(records)

def main():
    args = parse_args()
    raw_path = args.raw
    out_base = args.out
    x_mode = args.x_mode
    methods_filter = (
        [m.strip() for m in args.methods.split(',')] if args.methods else None
    )
    sources_filter = (
        [s.strip() for s in args.sources.split(',')] if args.sources else None
    )

    graphs_dir = os.path.join(out_base, "graphs")
    summary_dir = os.path.join(out_base, "summary")
    ensure_dir_exists(graphs_dir)
    ensure_dir_exists(summary_dir)

    df, file_info = load_all_data(raw_path, methods_filter, sources_filter)

    # Always plot *_vs_time.png for compatibility
    for metric in METRICS:
        plot_time_series_vs_time(df, metric, graphs_dir)

    # Plot time series with selected x_mode
    for metric in METRICS:
        plot_time_series(df, metric, graphs_dir, x_mode=x_mode)

    # Bar charts
    for metric in METRICS:
        plot_bar_summary(df, metric, graphs_dir)

    # Histograms and boxplots for latency_ms and fps_inst
    for metric in ['latency_ms', 'fps_inst']:
        plot_histogram(df, metric, graphs_dir)
        plot_boxplot(df, metric, graphs_dir)

    # Compute and save summary statistics
    summary_df = compute_summary_stats(df, file_info)
    summary_csv_path = os.path.join(summary_dir, "resumen_metricas.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Saved summary statistics to {summary_csv_path}")

if __name__ == "__main__":
    main()