#!/usr/bin/env python3
"""
Standardize and unify Section 4 metrics CSVs.

Usage:
    python Metrics/unify_section4_metrics.py \\
        --keypoints path/to/keypoints.csv \\
        --bboxes path/to/bboxes.csv \\
        --flower path/to/flower.csv \\
        [--label-keypoints LABEL] [--label-bboxes LABEL] [--label-flower LABEL] \\
        [--out OUTDIR]
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np

SECTION4_COLS = [
    "method",
    "frame_id",
    "elapsed_sec",
    "latency_ms",
    "fps_inst",
    "cpu_pct",
    "ram_mb",
    "detections"
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Standardize and unify Section 4 metrics CSVs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--keypoints", required=True, type=str, help="Path to KeyPoints CSV")
    parser.add_argument("--bboxes", required=True, type=str, help="Path to BBoxes CSV")
    parser.add_argument("--flower", required=True, type=str, help="Path to FlowerAI CSV")
    parser.add_argument("--label-keypoints", default="KeyPoints", type=str, help="Custom label for KeyPoints method")
    parser.add_argument("--label-bboxes", default="BBoxes", type=str, help="Custom label for BBoxes method")
    parser.add_argument("--label-flower", default="FlowerAI", type=str, help="Custom label for FlowerAI method")
    parser.add_argument("--out", default="Metrics/out/", type=str, help="Output directory")
    return parser.parse_args()


def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def process_keypoints_or_bboxes(path, label, label_override=False):
    # Read input, let pandas auto-detect delimiter/decimal
    df = pd.read_csv(path)
    original_shape = df.shape
    mapping = {}

    # Rename frame_idx to frame_id (int)
    if "frame_idx" in df.columns:
        df = df.rename(columns={"frame_idx": "frame_id"})
        mapping["frame_idx"] = "frame_id"
    else:
        raise ValueError(f"'frame_idx' column not found in {path}")

    # Create elapsed_sec column with NaN
    df["elapsed_sec"] = np.nan
    mapping["elapsed_sec"] = "created (filled NaN)"

    # Drop timestamp and source
    for col in ["timestamp", "source"]:
        if col in df.columns:
            df = df.drop(columns=[col])
            mapping[col] = "dropped"

    # Set method column value if label_override requested (else, keep original)
    if label_override:
        df["method"] = label
        mapping["method"] = f"set to '{label}'"
    else:
        mapping["method"] = "kept original"

    # Ensure types
    float_cols = ["latency_ms", "fps_inst", "cpu_pct", "ram_mb"]
    for c in float_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    int_cols = ["frame_id", "detections"]
    for c in int_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    # Add any missing columns with NaN
    for col in SECTION4_COLS:
        if col not in df.columns:
            df[col] = np.nan

    # Reorder columns
    df = df[SECTION4_COLS]

    return df, original_shape, mapping


def process_flower(path, label):
    df = pd.read_csv(path)
    original_shape = df.shape
    mapping = {}

    # Rename columns as specified
    rename_map = {
        "elapsed_s": "elapsed_sec",
        "cpu_percent": "cpu_pct",
        "rss_mb": "ram_mb",
        "cam_fps": "fps_inst",
        "cam_det": "detections"
    }
    df = df.rename(columns=rename_map)
    mapping.update(rename_map)

    # latency_ms = 1000.0 / fps_inst if > 0 else NaN
    def compute_latency(row):
        try:
            fps = float(row["fps_inst"])
            return 1000.0 / fps if fps > 0 else np.nan
        except Exception:
            return np.nan

    df["latency_ms"] = df.apply(compute_latency, axis=1)
    mapping["latency_ms"] = "computed as 1000.0/fps_inst"

    # frame_id as incremental index 0..N-1 (int)
    df["frame_id"] = pd.Series(range(len(df)), dtype="Int64")
    mapping["frame_id"] = "created incremental 0..N-1"

    # method column set to label
    df["method"] = label
    mapping["method"] = f"set to '{label}'"

    # Drop timestamp_iso, fit_calls, eval_calls
    drop_cols = [c for c in ["timestamp_iso", "fit_calls", "eval_calls"] if c in df.columns]
    df = df.drop(columns=drop_cols)
    for col in drop_cols:
        mapping[col] = "dropped"

    # Ensure types
    float_cols = ["elapsed_sec", "latency_ms", "fps_inst", "cpu_pct", "ram_mb"]
    for c in float_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    if "detections" in df.columns:
        df["detections"] = pd.to_numeric(df["detections"], errors="coerce").astype("Int64")
    if "frame_id" in df.columns:
        df["frame_id"] = pd.to_numeric(df["frame_id"], errors="coerce").astype("Int64")

    # Add any missing columns with NaN
    for col in SECTION4_COLS:
        if col not in df.columns:
            df[col] = np.nan

    # Reorder columns
    df = df[SECTION4_COLS]

    return df, original_shape, mapping


def main():
    args = parse_args()

    out_dir = os.path.abspath(args.out)
    std_dir = os.path.join(out_dir, "standardized")
    ensure_dir_exists(std_dir)

    # 1. KeyPoints
    kp_df, kp_shape, kp_map = process_keypoints_or_bboxes(
        args.keypoints, args.label_keypoints,
        label_override=(args.label_keypoints != "KeyPoints")
    )
    kp_std_path = os.path.join(std_dir, "keypoints_std.csv")
    kp_df.to_csv(kp_std_path, index=False)
    print(f"[KeyPoints] {kp_shape[0]} rows processed. Mapping: {kp_map}")
    print(f"    Output: {kp_std_path}")

    # 2. BBoxes
    bb_df, bb_shape, bb_map = process_keypoints_or_bboxes(
        args.bboxes, args.label_bboxes,
        label_override=(args.label_bboxes != "BBoxes")
    )
    bb_std_path = os.path.join(std_dir, "bboxes_std.csv")
    bb_df.to_csv(bb_std_path, index=False)
    print(f"[BBoxes] {bb_shape[0]} rows processed. Mapping: {bb_map}")
    print(f"    Output: {bb_std_path}")

    # 3. FlowerAI
    flower_df, flower_shape, flower_map = process_flower(args.flower, args.label_flower)
    flower_std_path = os.path.join(std_dir, "flower_std.csv")
    flower_df.to_csv(flower_std_path, index=False)
    print(f"[FlowerAI] {flower_shape[0]} rows processed. Mapping: {flower_map}")
    print(f"    Output: {flower_std_path}")

    # Concatenate all, sort by method then frame_id
    all_df = pd.concat([kp_df, bb_df, flower_df], ignore_index=True)
    all_df = all_df.sort_values(by=["method", "frame_id"]).reset_index(drop=True)
    all_csv_path = os.path.join(out_dir, "section4_metrics_all.csv")
    all_df.to_csv(all_csv_path, index=False)
    print(f"[ALL] {all_df.shape[0]} rows combined and written to {all_csv_path}")


if __name__ == "__main__":
    main()