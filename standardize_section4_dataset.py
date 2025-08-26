#!/usr/bin/env python3

import argparse
import json
import sys
import shutil
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, List

CANONICAL_COLUMNS = [
    "frame_id", "session_id", "method", "mode", "source", "device", "threshold", "elapsed_sec",
    "latency_ms", "fps_inst", "cpu_pct", "ram_mb", "detections", "detection_flag", "presence_flag",
    "fit_calls", "eval_calls"
]

KEY_METRICS = [
    "latency_ms", "fps_inst", "cpu_pct", "ram_mb", "detection_flag", "presence_flag", "threshold"
]

FLOWER_COL_MAPPING = {
    # common misspellings and flower-specific columns mapped to canonical names
    "frame": "frame_id",
    "frame_no": "frame_id",
    "frame_number": "frame_id",
    "session": "session_id",
    "sessionid": "session_id",
    "elapsed_time": "elapsed_sec",
    "elapsed_s": "elapsed_sec",
    "latency": "latency_ms",
    "latency (ms)": "latency_ms",
    "latency_ms": "latency_ms",
    "fps": "fps_inst",
    "cpu%": "cpu_pct",
    "cpu_percent": "cpu_pct",
    "cpu": "cpu_pct",
    "ram": "ram_mb",
    "ram_mb": "ram_mb",
    "detections": "detections",
    "detection": "detections",
    "detection_flag": "detection_flag",
    "det_flag": "detection_flag",
    "presence_flag": "presence_flag",
    "fit calls": "fit_calls",
    "fit_calls": "fit_calls",
    "eval_calls": "eval_calls",
    "threshold": "threshold",
    "method": "method",
    "mode": "mode",
    "device": "device",
}

METHODS = ["yolo", "ssd", "mobilenet", "flowerai", "retinanet", "fasterrcnn", "rcnn", "custom"]
MODES = ["live", "offline", "batch", "video", "image", "realtime"]

def parse_args():
    parser = argparse.ArgumentParser(description="Standardize Section 4 Dataset CSVs.")
    parser.add_argument('--raw', type=str, required=True, help='Input directory with raw CSV files')
    parser.add_argument('--out', type=str, required=True, help='Output directory')
    parser.add_argument('--default-threshold', type=float, default=0.5, help='Default detection threshold')
    parser.add_argument('--presence-mode', type=str, default='column', choices=['column', 'ranges'], help='How to determine presence_flag')
    parser.add_argument('--presence-column', type=str, default='presence_flag', help='Column to use for presence if presence-mode=column')
    parser.add_argument('--presence-ranges', type=str, default=None, help='JSON or YAML file specifying presence ranges (for presence-mode=ranges)')
    return parser.parse_args()

def load_presence_ranges(path: Optional[str]) -> Dict[str, List[Tuple[int, int]]]:
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Presence ranges file not found: {p}")
    if p.suffix.lower() in ['.json']:
        with open(p, 'r') as f:
            return json.load(f)
    elif p.suffix.lower() in ['.yaml', '.yml']:
        with open(p, 'r') as f:
            return yaml.safe_load(f)
    else:
        raise ValueError(f"Unknown presence-ranges file extension: {p.suffix}")

def normalize_colname(name: str) -> str:
    return name.strip().lower().replace(" ", "_")

def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Lowercase, strip, replace spaces with underscores
    df = df.rename(columns={c: normalize_colname(c) for c in df.columns})
    # Map flower-specific columns to canonical names
    df = df.rename(columns={k: v for k, v in FLOWER_COL_MAPPING.items() if k in df.columns})
    return df

def infer_method_mode(filename: str) -> Tuple[Optional[str], Optional[str]]:
    lower = filename.lower()
    method = None
    for m in METHODS:
        if m in lower:
            method = m
            break
    mode = None
    for md in MODES:
        if md in lower:
            mode = md
            break
    return method, mode

def infer_session_id(filename: str) -> str:
    # Use stem as session_id
    return Path(filename).stem

def derive_frame_id(df: pd.DataFrame) -> pd.DataFrame:
    if "frame_id" not in df.columns or df["frame_id"].isnull().all():
        df["frame_id"] = np.arange(len(df))
        print("Derived 'frame_id' as row number (0-based)")
    return df

def derive_elapsed_sec(df: pd.DataFrame) -> pd.DataFrame:
    if "elapsed_sec" not in df.columns or df["elapsed_sec"].isnull().all():
        if "latency_ms" in df.columns and "fps_inst" in df.columns:
            print("Deriving 'elapsed_sec' from frame_id and fps_inst (approximate, cumulative)")
            # Approximate, cumulative sum of frame times
            df["elapsed_sec"] = np.cumsum(1 / df["fps_inst"].replace(0, np.nan).fillna(1))
        else:
            df["elapsed_sec"] = df.index  # fallback
            print("Derived 'elapsed_sec' as row number (fallback)")
    return df

def derive_latency_or_fps(df: pd.DataFrame) -> pd.DataFrame:
    # If one is present but not the other, try to derive
    if "latency_ms" in df.columns and "fps_inst" not in df.columns:
        print("Deriving 'fps_inst' from 'latency_ms'")
        df["fps_inst"] = 1000.0 / df["latency_ms"].replace(0, np.nan)
        df["fps_inst"] = df["fps_inst"].replace([np.inf, -np.inf], 0).fillna(0)
    elif "fps_inst" in df.columns and "latency_ms" not in df.columns:
        print("Deriving 'latency_ms' from 'fps_inst'")
        df["latency_ms"] = 1000.0 / df["fps_inst"].replace(0, np.nan)
        df["latency_ms"] = df["latency_ms"].replace([np.inf, -np.inf], 0).fillna(0)
    return df

def add_missing_columns(df: pd.DataFrame, default_row: Dict = None) -> pd.DataFrame:
    if default_row is None:
        default_row = {}
    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            default_val = default_row.get(col, 0 if col in KEY_METRICS else np.nan)
            df[col] = default_val
    return df

def fill_nans(df: pd.DataFrame, cols: List[str]):
    for col in cols:
        if col in df.columns:
            n_nans = df[col].isna().sum()
            if n_nans > 0:
                print(f"Filling {n_nans} NaNs in '{col}' with 0")
                df[col] = df[col].fillna(0)
    return df

def set_device_and_source(df: pd.DataFrame, filename: str):
    df["device"] = "RPi"
    df["source"] = Path(filename).name
    return df

def set_threshold(df: pd.DataFrame, default_threshold: float):
    if "threshold" in df.columns:
        n_missing = df["threshold"].isna().sum()
        if n_missing > 0:
            print(f"Filling {n_missing} missing thresholds with default {default_threshold}")
            df["threshold"] = df["threshold"].fillna(default_threshold)
    else:
        df["threshold"] = default_threshold
        print(f"Set all threshold to default {default_threshold}")
    return df

def set_presence_flag(df: pd.DataFrame, presence_mode: str, presence_column: str, presence_ranges: Dict, session_id: str):
    if presence_mode == "column":
        if presence_column in df.columns:
            # ensure binary 0/1
            df["presence_flag"] = df[presence_column].astype(int)
        else:
            print(f"Warning: presence column '{presence_column}' not found, setting all presence_flag=0")
            df["presence_flag"] = 0
    elif presence_mode == "ranges":
        # presence_ranges: dict of session_id -> list of [start, end] frame_id pairs
        flags = np.zeros(len(df), dtype=int)
        ranges = presence_ranges.get(session_id, [])
        for start, end in ranges:
            # inclusive range
            mask = (df["frame_id"] >= start) & (df["frame_id"] <= end)
            flags[mask.values] = 1
        df["presence_flag"] = flags
    else:
        raise ValueError(f"Unknown presence_mode: {presence_mode}")
    return df

def check_for_key_nans(df: pd.DataFrame, filename: str):
    nans = {}
    for col in KEY_METRICS:
        if col in df.columns:
            n = df[col].isna().sum()
            if n > 0:
                nans[col] = n
    if nans:
        print(f"Warning: {filename} has NaNs in {nans}")
    return nans

def process_file(
    file: Path,
    out_dir: Path,
    default_threshold: float,
    presence_mode: str,
    presence_column: str,
    presence_ranges: Dict
) -> Optional[Path]:
    print(f"\nProcessing {file}...")
    try:
        df = pd.read_csv(file)
    except Exception as e:
        print(f"Failed to read {file}: {e}")
        return None

    df = canonicalize_columns(df)
    method, mode = infer_method_mode(file.name)
    if method is None or mode is None:
        print(f"Warning: Could not infer method/mode from filename {file.name}, skipping.")
        return None
    session_id = infer_session_id(file.name)
    df["session_id"] = session_id
    df["method"] = method
    df["mode"] = mode

    df = derive_frame_id(df)
    df = derive_elapsed_sec(df)
    df = derive_latency_or_fps(df)
    df = set_device_and_source(df, file.name)
    df = set_threshold(df, default_threshold)
    df = add_missing_columns(df)
    df = set_presence_flag(df, presence_mode, presence_column, presence_ranges, session_id)
    df = fill_nans(df, KEY_METRICS)
    check_for_key_nans(df, file.name)

    # Reorder columns
    df = df[[col for col in CANONICAL_COLUMNS if col in df.columns] + [c for c in df.columns if c not in CANONICAL_COLUMNS]]

    # Output
    std_dir = out_dir / "standardized"
    std_dir.mkdir(parents=True, exist_ok=True)
    out_file = std_dir / f"{file.stem}_std.csv"
    df.to_csv(out_file, index=False)
    print(f"Standardized file written to {out_file}")
    return out_file

def aggregate_sessions(df: pd.DataFrame) -> pd.DataFrame:
    # Group by session_id, method, mode
    def agg_coverage(subdf):
        numer = ((subdf["detection_flag"] == 1) & (subdf["presence_flag"] == 1)).sum()
        denom = subdf["presence_flag"].sum()
        return numer / max(1, denom)
    agg_dict = {
        "frame_id": "count",
        "fps_inst": "mean",
        "latency_ms": ["mean", lambda x: np.percentile(x, 95)],
        "cpu_pct": ["mean", lambda x: np.percentile(x, 95), "max"],
        "ram_mb": ["mean", lambda x: np.percentile(x, 95), "max"],
        "detection_flag": "mean",
        "threshold": lambda x: x.value_counts().index[0] if len(x.value_counts()) > 0 else np.nan,
        "elapsed_sec": [lambda x: x.max() - x.min()],
        "presence_flag": "sum",
        "fit_calls": "sum",
        "eval_calls": "sum"
    }
    grouped = df.groupby(["session_id", "method", "mode"])
    agg = grouped.agg(agg_dict)
    # Flatten columns
    agg.columns = [
        "n_frames", "fps_mean", "lat_mean_ms", "lat_p95_ms",
        "cpu_mean", "cpu_p95", "cpu_peak",
        "ram_mean_mb", "ram_p95_mb", "ram_peak_mb",
        "detection_rate", "threshold_used", "duration_sec",
        "presence_count", "fit_calls", "eval_calls"
    ]
    # Compute coverage
    agg["coverage"] = grouped.apply(agg_coverage)
    agg = agg.reset_index()
    return agg

def generate_overview(
    sessions_df: pd.DataFrame,
    all_df: pd.DataFrame,
    out_dir: Path,
    processed_sessions: List[str],
    file_stats: Dict[str, Dict]
):
    lines = []
    lines.append("# Section 4 Dataset Overview\n")
    lines.append("## Processed Sessions:\n")
    for sess in processed_sessions:
        lines.append(f"- {sess}")
    lines.append("\n## Key Metrics NaN Verification:")
    any_nans = False
    for src, stats in file_stats.items():
        if stats:
            lines.append(f"  - {src}: {stats}")
            any_nans = True
    if not any_nans:
        lines.append("  - No NaNs in key metrics in any standardized file.")

    lines.append("\n## Per-Method Stats:")
    for method in all_df["method"].unique():
        mdf = all_df[all_df["method"] == method]
        lines.append(f"\n### {method}")
        lines.append(f"  - Latency (ms): mean {mdf['latency_ms'].mean():.2f}, p95 {np.percentile(mdf['latency_ms'],95):.2f}, max {mdf['latency_ms'].max():.2f}")
        lines.append(f"  - FPS: mean {mdf['fps_inst'].mean():.2f}, p95 {np.percentile(mdf['fps_inst'],95):.2f}, max {mdf['fps_inst'].max():.2f}")
        lines.append(f"  - CPU (%): mean {mdf['cpu_pct'].mean():.2f}, p95 {np.percentile(mdf['cpu_pct'],95):.2f}, max {mdf['cpu_pct'].max():.2f}")
        lines.append(f"  - RAM (MB): mean {mdf['ram_mb'].mean():.2f}, p95 {np.percentile(mdf['ram_mb'],95):.2f}, max {mdf['ram_mb'].max():.2f}")

    lines.append("\n## Presence Flag Count Per Session:")
    for sess in sessions_df["session_id"].unique():
        n_pres = sessions_df[sessions_df["session_id"]==sess]["presence_count"].values[0]
        lines.append(f"  - {sess}: {n_pres}")

    flower_rows = all_df.shape[0]
    flower_cpu_mean = all_df['cpu_pct'].mean()
    flower_ram_mean = all_df['ram_mb'].mean()
    lines.append(f"\n## Special Note:\n- Flower rows count: {flower_rows}")
    lines.append(f"- Mean CPU: {flower_cpu_mean:.2f}%, Mean RAM: {flower_ram_mean:.2f} MB")

    overview_path = out_dir / "summary" / "section4_overview.txt"
    overview_path.parent.mkdir(parents=True, exist_ok=True)
    with open(overview_path, "w") as f:
        f.write('\n'.join(lines))
    print(f"Overview report written to {overview_path}")

def main():
    args = parse_args()
    raw_dir = Path(args.raw).resolve()
    out_dir = Path(args.out).resolve()
    default_threshold = args.default_threshold
    presence_mode = args.presence_mode
    presence_column = args.presence_column
    presence_ranges = load_presence_ranges(args.presence_ranges) if presence_mode == "ranges" else {}

    # Prepare output directories
    std_dir = out_dir / "standardized"
    summary_dir = out_dir / "summary"
    for d in [std_dir, summary_dir]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    # Walk all CSVs in raw_dir
    csvs = sorted([p for p in raw_dir.rglob("*.csv") if p.is_file()])
    print(f"Found {len(csvs)} CSV files in {raw_dir}")

    standardized_files = []
    processed_sessions = []
    file_stats = {}

    for csv in csvs:
        out_file = process_file(
            csv, out_dir, default_threshold, presence_mode, presence_column, presence_ranges
        )
        if out_file is not None:
            standardized_files.append(out_file)
            processed_sessions.append(Path(csv).stem)
            # Re-check NaNs for reporting
            df = pd.read_csv(out_file)
            file_stats[str(out_file)] = check_for_key_nans(df, str(out_file))
        else:
            print(f"Skipped file: {csv}")

    # Concatenate all standardized files
    if not standardized_files:
        print("No standardized files produced. Exiting.", file=sys.stderr)
        sys.exit(1)

    all_df = pd.concat([pd.read_csv(f) for f in standardized_files], ignore_index=True)
    all_df.to_csv(summary_dir / "section4_dataset.csv", index=False)
    print(f"Concatenated dataset written to {summary_dir / 'section4_dataset.csv'}")

    # Group and aggregate
    sessions_df = aggregate_sessions(all_df)
    sessions_df.to_csv(summary_dir / "section4_sessions.csv", index=False)
    print(f"Session aggregates written to {summary_dir / 'section4_sessions.csv'}")

    # Overview report
    generate_overview(sessions_df, all_df, out_dir, processed_sessions, file_stats)


if __name__ == "__main__":
    main()