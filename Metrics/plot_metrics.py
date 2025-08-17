import argparse
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

METRICS = ['latency_ms', 'fps_inst', 'cpu_pct', 'ram_mb']
SUMMARY_STATS = ['mean', 'median', 'p95', 'std', 'count']

def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def parse_args():
    parser = argparse.ArgumentParser(description="Plot and summarize metrics")
    parser.add_argument('--raw', type=str, required=True, help='Input directory with raw CSVs')
    parser.add_argument('--out', type=str, required=True, help='Base output directory')
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

def fill_missing_columns(df):
    # Ensure all metrics columns exist, fill with NaN if missing
    for col in METRICS:
        if col not in df.columns:
            df[col] = np.nan
    # Compute fps_inst if missing
    if 'fps_inst' not in df.columns or df['fps_inst'].isnull().all():
        # If latency_ms exists, compute fps_inst if possible
        if 'latency_ms' in df.columns and not df['latency_ms'].isnull().all():
            df['fps_inst'] = 1000.0 / df['latency_ms']
    return df

def load_all_data(raw_path):
    files = get_csv_files(raw_path)
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            # Add 'method' column if missing (try to infer from filename)
            if 'method' not in df.columns:
                method_name = os.path.splitext(os.path.basename(f))[0]
                df['method'] = method_name
            df = fill_missing_columns(df)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
    if not dfs:
        raise RuntimeError("No valid CSV files found in input directory.")
    big_df = pd.concat(dfs, ignore_index=True)
    return big_df

def plot_time_series(df, metric, graphs_dir):
    plt.figure(figsize=(10, 6))
    for method, group in df.groupby('method'):
        # Sort by timestamp if exists, else by index
        if 'timestamp' in group.columns:
            group = group.sort_values('timestamp')
            x = pd.to_datetime(group['timestamp'], errors='coerce')
        else:
            x = range(len(group))
        plt.plot(x, group[metric], label=method)
    plt.xlabel('Timestamp' if 'timestamp' in df.columns else 'Sample')
    plt.ylabel(metric)
    plt.title(f"{metric} vs Time")
    plt.legend()
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
    plt.ylabel(metric)
    plt.title(f"Mean ± Std of {metric} per Method")
    plt.tight_layout()
    out_path = os.path.join(graphs_dir, f"resumen_{metric}.png")
    plt.savefig(out_path)
    plt.close()

def compute_summary_stats(df):
    # p95 requires custom lambda
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
        records.append(rec)
    return pd.DataFrame.from_records(records)

def main():
    args = parse_args()
    raw_path = args.raw
    out_base = args.out

    graphs_dir = os.path.join(out_base, "graphs")
    summary_dir = os.path.join(out_base, "summary")
    ensure_dir_exists(graphs_dir)
    ensure_dir_exists(summary_dir)

    df = load_all_data(raw_path)

    # Plot time series and bar summaries for each metric
    for metric in METRICS:
        plot_time_series(df, metric, graphs_dir)
        plot_bar_summary(df, metric, graphs_dir)

    # Compute and save summary statistics
    summary_df = compute_summary_stats(df)
    summary_csv_path = os.path.join(summary_dir, "resumen_metricas.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Saved summary statistics to {summary_csv_path}")

if __name__ == "__main__":
    main()