import argparse
import os
import pandas as pd
from shutil import copyfile
import logging

def parse_args():
    parser = argparse.ArgumentParser(description="Build coordinated datasets based on FlowerAI detection.")
    parser.add_argument('--input', type=str, default='Metrics/out/section4_metrics_all.csv',
                        help='Path to metrics CSV file')
    parser.add_argument('--out', type=str, default='Metrics/out/standardized/',
                        help='Output directory for coordinated CSVs')
    parser.add_argument('--min-dets', type=int, default=1,
                        help='Minimum number of detections for flower presence')
    parser.add_argument('--methods', type=str, default="BBoxes-YOLOv4tiny,KeyPoints-ResNet50,FlowerAI",
                        help='Comma-separated list of methods')
    return parser.parse_args()

def ensure_output_dir(out_path):
    os.makedirs(out_path, exist_ok=True)

def normalize_columns(df):
    df.columns = [c.strip().lower() for c in df.columns]
    orig_cols = list(df.columns)
    # Only rename elapsed_sec or id_elapsed_sec to frame_id if frame_id does not already exist
    if 'frame_id' not in df.columns:
        if 'id_elapsed_sec' in df.columns:
            df = df.rename(columns={'id_elapsed_sec': 'frame_id'})
        elif 'elapsed_sec' in df.columns:
            df = df.rename(columns={'elapsed_sec': 'frame_id'})
    # Remove duplicate columns, keep first occurrence
    before = len(df.columns)
    df = df.loc[:, ~df.columns.duplicated()]
    after = len(df.columns)
    dups_removed = before - after
    if dups_removed > 0:
        logging.info(f"Removed {dups_removed} duplicate columns after renaming in normalize_columns.")
    return df

def load_csv(csv_path):
    df = pd.read_csv(csv_path)
    df = normalize_columns(df)
    # Ensure frame_id is numeric (coerce errors)
    if 'frame_id' in df.columns:
        df['frame_id'] = pd.to_numeric(df['frame_id'], errors='coerce')
    return df

def get_flower_present_frames(metrics_df, min_dets):
    # FlowerAI rows
    flower_df = metrics_df[metrics_df['method'] == 'FlowerAI']
    # Use detection_flag if present, else detections
    if 'detection_flag' in flower_df.columns:
        flower_present = flower_df['detection_flag'] >= min_dets
    elif 'detections' in flower_df.columns:
        flower_present = flower_df['detections'] >= min_dets
    else:
        raise ValueError("Neither 'detection_flag' nor 'detections' column found in FlowerAI rows.")
    # Build set of unique ints, dropna, astype(int)
    present_frames = set(
        flower_df.loc[flower_present, 'frame_id'].dropna().astype(int).unique()
    )
    return present_frames

def method_to_prefix(method):
    mapping = {
        "BBoxes": "bboxes",
        "KeyPoints": "keypoints",
        "FlowerAI": "flower"
    }
    # Method e.g. "BBoxes-YOLOv4tiny" -> "BBoxes"
    key = method.split('-')[0]
    return mapping.get(key, key.lower())

def main():
    logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)
    args = parse_args()

    ensure_output_dir(args.out)

    # Load metrics file and normalize
    metrics_df = load_csv(args.input)
    methods = [m.strip() for m in args.methods.split(',') if m.strip()]

    # Make sure frame_id is present
    if 'frame_id' not in metrics_df.columns:
        raise ValueError("No 'frame_id', 'elapsed_sec', or 'id_elapsed_sec' column in input metrics CSV.")

    # Get flower-present frames
    flower_present_frames = get_flower_present_frames(metrics_df, args.min_dets)

    # If present_ids is empty, log a warning and copy baseline to coord for each method
    if not flower_present_frames:
        logging.warning("No present frames found in FlowerAI metrics (present_ids is empty).")
        for method in methods:
            prefix = method_to_prefix(method)
            baseline_path = os.path.join(args.out, f"{prefix}_std.csv")
            coord_path = os.path.join(args.out, f"{prefix}_std_coord.csv")
            if not os.path.isfile(baseline_path):
                logging.warning(f"Baseline file {baseline_path} not found for method '{method}'. Skipping.")
                continue
            copyfile(baseline_path, coord_path)
            logging.warning(f"[{method}] present_ids empty; copied baseline file to coord path without filtering.")
        return

    for method in methods:
        prefix = method_to_prefix(method)
        baseline_path = os.path.join(args.out, f"{prefix}_std.csv")
        coord_path = os.path.join(args.out, f"{prefix}_std_coord.csv")
        if not os.path.isfile(baseline_path):
            logging.warning(f"Baseline file {baseline_path} not found for method '{method}'. Skipping.")
            continue

        # Special case: FlowerAI
        if prefix == "flower":
            df = load_csv(baseline_path)
            orig_count = len(df)
            # Use robust Int64 logic for filtering
            if 'frame_id' in df.columns:
                mask = df['frame_id'].astype('Int64').isin(flower_present_frames)
                df_coord = df.loc[mask]
            else:
                df_coord = df.iloc[[]]
            coord_count = len(df_coord)
            df_coord.to_csv(coord_path, index=False)
            logging.info(f"[{method}] baseline: {orig_count}, coord: {coord_count}")
            continue

        # For other methods
        df = load_csv(baseline_path)
        orig_count = len(df)
        if 'frame_id' not in df.columns:
            logging.warning(f"File {baseline_path} does not have a 'frame_id' column after normalization.")
            continue
        mask = df['frame_id'].astype('Int64').isin(flower_present_frames)
        df_coord = df.loc[mask]
        coord_count = len(df_coord)
        df_coord.to_csv(coord_path, index=False)
        logging.info(f"[{method}] baseline: {orig_count}, coord: {coord_count}")

if __name__ == '__main__':
    main()