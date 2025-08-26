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
    # Rename elapsed_sec or id_elapsed_sec to frame_id if present
    if 'id_elapsed_sec' in df.columns:
        df = df.rename(columns={'id_elapsed_sec': 'frame_id'})
    elif 'elapsed_sec' in df.columns:
        df = df.rename(columns={'elapsed_sec': 'frame_id'})
    return df

def load_csv(csv_path):
    df = pd.read_csv(csv_path)
    return normalize_columns(df)

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
    present_frames = set(flower_df.loc[flower_present, 'frame_id'])
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

    for method in methods:
        prefix = method_to_prefix(method)
        baseline_path = os.path.join(args.out, f"{prefix}_std.csv")
        coord_path = os.path.join(args.out, f"{prefix}_std_coord.csv")
        if not os.path.isfile(baseline_path):
            logging.warning(f"Baseline file {baseline_path} not found for method '{method}'. Skipping.")
            continue

        # Special case: FlowerAI
        if prefix == "flower":
            # If the standardized file exists, just filter to frames (if needed) and save as coord file
            df = load_csv(baseline_path)
            orig_count = len(df)
            df_coord = df[df['frame_id'].isin(flower_present_frames)]
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
        df_coord = df[df['frame_id'].isin(flower_present_frames)]
        coord_count = len(df_coord)
        df_coord.to_csv(coord_path, index=False)
        logging.info(f"[{method}] baseline: {orig_count}, coord: {coord_count}")

if __name__ == '__main__':
    main()