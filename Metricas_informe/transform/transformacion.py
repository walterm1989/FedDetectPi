import argparse
import glob
import os
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd


STANDARD_COLUMNS = [
    "timestamp",
    "fps",
    "latency_ms",
    "cpu_percent",
    "ram_mb",
    "method",
    "model",
    "scope",
    "file_source",
    "frame_idx",
]

SYNONYMS: Dict[str, List[str]] = {
    "timestamp": ["timestamp", "time", "datetime", "frame_time", "ts"],
    "fps": ["fps", "FPS", "frames_per_second", "fps_inst"],
    "latency_ms": ["latency_ms", "latency", "lat_ms", "inference_ms", "inference_time_ms", "latencia_ms"],
    "cpu_percent": ["cpu_percent", "cpu", "CPU_%", "cpu_usage", "cpu_use", "cpu_pct"],
    "ram_mb": ["ram_mb", "memory_mb", "mem_mb", "ram_usage_mb", "rss_mb", "memory"],
    "frame_idx": ["frame", "frame_idx", "index"],
    "method": ["method"],  # will be captured as raw_method
}

CONSOLIDATED_BASENAME = "metrica_consolidada.csv"


def discover_csvs(raw_dir: str, out_path: str) -> List[str]:
    # Discover recursively, exclude the consolidated file and any file that looks generated like temporary files.
    all_csvs = glob.glob(os.path.join(raw_dir, "**", "*.csv"), recursive=True)
    out_abs = os.path.abspath(out_path)
    results = []
    for p in sorted(all_csvs):
        if os.path.abspath(p) == out_abs:
            continue
        if os.path.basename(p) == CONSOLIDATED_BASENAME:
            continue
        # Exclude common temp artifacts
        base = os.path.basename(p)
        if base.startswith("~$") or base.endswith(".tmp.csv"):
            continue
        results.append(p)
    return results


def scope_from_path(path: str) -> str:
    low = path.replace("\\", "/").lower()
    if "/flowerai/" in low:
        return "flower"
    if "/local/" in low:
        return "local"
    # Default: try parent folder name
    parts = low.split("/")
    if "flowerai" in parts:
        return "flower"
    if "local" in parts:
        return "local"
    return "unknown"


def parse_method_model(raw_value: Optional[str], filename: str) -> Tuple[str, str]:
    """
    raw_value may be a string like "BBoxes-YOLOv4tiny" or "KeyPoints-resnet50".
    If not present, infer from filename using same rules.
    """
    candidate = (raw_value or "").strip()
    if not candidate:
        candidate = os.path.splitext(os.path.basename(filename))[0]

    cand_low = candidate.lower()

    if "bboxes" in cand_low or "bbox" in cand_low or "boundingboxes" in cand_low:
        method = "BoundingBoxes"
    elif "keypoints" in cand_low or "kpts" in cand_low or "key_points" in cand_low:
        method = "KeyPoints"
    else:
        method = "unknown"

    # Try to extract model after "-" or known tokens
    model = "unknown"
    # Tokens split
    tokens = re.split(r"[_\-\s]+", candidate)
    # Try to find a token that looks like model
    for tok in tokens:
        tl = tok.lower()
        if not tl:
            continue
        if tl.startswith("yolo"):
            # Keep original token case for YOLO variants if present in original
            # Prefer the original mixed case from candidate by searching
            m = re.search(r"(YOLO[^\s_\-]+)", candidate)
            model = m.group(1) if m else tok
            break
        if tl in {"resnet50", "resnet-50", "resnet_50"}:
            model = "ResNet50"
            break
        if tl.startswith("resnet"):
            # Normalize variants like resnet34, resnet101 if ever seen
            digits = re.findall(r"\d+", tl)
            model = f"ResNet{digits[0]}" if digits else "ResNet"
            break

    return method, model


def coerce_cpu_percent(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace("%", "", regex=False).str.replace(",", ".", regex=False).str.strip()
    s = pd.to_numeric(s, errors="coerce")
    # If values look like 0-1 range, scale
    if s.notna().any():
        finite = s.dropna()
        if len(finite) and finite.max() <= 1.5:
            s = s * 100.0
    s = s.clip(lower=0, upper=100)
    return s


def parse_timestamp(col: pd.Series) -> pd.Series:
    # Try direct parse
    ts = pd.to_datetime(col, errors="coerce", utc=True, infer_datetime_format=True)
    # If still many NaT and values numeric, try epoch heuristics
    needs = ts.isna()
    if needs.any():
        # Try numeric conversion
        nums = pd.to_numeric(col[needs], errors="coerce")
        if nums.notna().any():
            # Heuristic by magnitude
            # microseconds
            mask = nums >= 1e15
            if mask.any():
                ts.loc[needs[needs].index[mask]] = pd.to_datetime(nums[mask], unit="us", utc=True)
            # milliseconds
            mask = (nums >= 1e12) & (nums < 1e15)
            if mask.any():
                ts.loc[needs[needs].index[mask]] = pd.to_datetime(nums[mask], unit="ms", utc=True)
            # seconds (unix epoch)
            mask = (nums >= 1e9) & (nums < 1e12)
            if mask.any():
                ts.loc[needs[needs].index[mask]] = pd.to_datetime(nums[mask], unit="s", utc=True)
    return ts


def extract_datetime_from_filename(path: str) -> Optional[pd.Timestamp]:
    base = os.path.basename(path)
    m = re.search(r"(\d{8})_(\d{6})", base)
    if not m:
        return None
    ymd = m.group(1)
    hms = m.group(2)
    try:
        dt = datetime.strptime(ymd + hms, "%Y%m%d%H%M%S")
        return pd.Timestamp(dt, tz="UTC")
    except Exception:
        return None


def standardize_dataframe(
    df: pd.DataFrame, src_path: str, raw_dir: str
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    # Record raw columns
    raw_columns = list(df.columns)

    # Map columns according to synonyms
    col_map: Dict[str, str] = {}  # src_col -> standard
    lower_cols = {c.lower(): c for c in df.columns}
    for std, aliases in SYNONYMS.items():
        for alias in aliases:
            if alias.lower() in lower_cols:
                col_map[lower_cols[alias.lower()]] = std
                break  # first match

    # Handle method specially, store as raw_method if present
    raw_method_val = None
    if "method" in col_map.values():
        # Find original column name mapped to method
        orig = [k for k, v in col_map.items() if v == "method"][0]
        # Copy it to raw_method and don't keep as standard "method"
        raw_method_val = df[orig].astype(str)
        # Remove mapping to avoid using raw "method" directly
        del col_map[orig]

    # Build working frame
    work = pd.DataFrame(index=df.index.copy())

    # timestamp handling
    if any(v == "timestamp" for v in col_map.values()):
        orig = [k for k, v in col_map.items() if v == "timestamp"][0]
        ts = parse_timestamp(df[orig])
    else:
        ts = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")

    # fps
    fps = None
    if any(v == "fps" for v in col_map.values()):
        orig = [k for k, v in col_map.items() if v == "fps"][0]
        fps = pd.to_numeric(df[orig], errors="coerce")

    # latency_ms
    latency = None
    if any(v == "latency_ms" for v in col_map.values()):
        orig = [k for k, v in col_map.items() if v == "latency_ms"][0]
        latency = pd.to_numeric(df[orig], errors="coerce")

    # Derive fps/latency if needed
    if fps is None and latency is not None:
        fps = 1000.0 / latency.replace(0, pd.NA)
    if latency is None and fps is not None:
        latency = 1000.0 / fps.replace(0, pd.NA)
    if fps is None and latency is None:
        raise ValueError("Neither fps nor latency_ms present or derivable.")

    # cpu_percent
    cpu = None
    if any(v == "cpu_percent" for v in col_map.values()):
        orig = [k for k, v in col_map.items() if v == "cpu_percent"][0]
        cpu = coerce_cpu_percent(df[orig]).fillna(0)
    else:
        # Default to 0 to satisfy "sin nulos" and valid range [0,100]
        cpu = pd.Series(0.0, index=df.index, dtype="float")

    # ram_mb
    ram = None
    if any(v == "ram_mb" for v in col_map.values()):
        orig = [k for k, v in col_map.items() if v == "ram_mb"][0]
        ram = pd.to_numeric(df[orig], errors="coerce").fillna(0)
    else:
        # Default to 0 MB if not present
        ram = pd.Series(0.0, index=df.index, dtype="float")

    # frame_idx
    frame_idx = None
    if any(v == "frame_idx" for v in col_map.values()):
        orig = [k for k, v in col_map.items() if v == "frame_idx"][0]
        frame_idx = pd.to_numeric(df[orig], errors="coerce").fillna(0).astype(int)
    else:
        frame_idx = pd.Series(range(len(df)), index=df.index, dtype="int")

    # Determine scope
    scope = scope_from_path(src_path)

    # Determine method/model
    raw_method_str = None
    if raw_method_val is not None:
        # If mixed values across rows, prefer first non-null
        first_non_null = raw_method_val.dropna()
        raw_method_str = first_non_null.iloc[0] if len(first_non_null) else None
    method, model = parse_method_model(raw_method_str, src_path)

    # Fill timestamps if missing
    if ts.isna().any():
        anchor = extract_datetime_from_filename(src_path)
        if anchor is None:
            # Deterministic anchor
            anchor = pd.Timestamp(datetime(2025, 1, 1, 0, 0, 0), tz="UTC")
        # spacing based on median fps or fallback 0.2s (5 fps)
        fps_median = None
        if fps is not None and pd.api.types.is_numeric_dtype(fps):
            finite = fps.replace([pd.NA, pd.NaT], pd.NA).dropna()
            if len(finite):
                fps_median = float(finite.median())
        step_seconds = 1.0 / fps_median if fps_median and fps_median > 0 else 0.2
        # Fill NaT with sequence based on frame_idx ordering
        # Ensure deterministic ordering by existing index order
        fill_series = []
        for i in range(len(ts)):
            fill_series.append(anchor + pd.Timedelta(seconds=step_seconds * i))
        ts_filled = pd.Series(fill_series, index=ts.index)
        ts = ts.fillna(ts_filled)

    # Assemble standardized frame
    work["timestamp"] = ts
    work["fps"] = pd.to_numeric(fps, errors="coerce")
    work["latency_ms"] = pd.to_numeric(latency, errors="coerce")
    work["cpu_percent"] = pd.to_numeric(cpu, errors="coerce")
    work["ram_mb"] = pd.to_numeric(ram, errors="coerce")
    work["method"] = method
    work["model"] = model
    work["scope"] = scope
    # file_source relative to raw_dir
    work["file_source"] = os.path.relpath(src_path, raw_dir).replace("\\", "/")
    work["frame_idx"] = frame_idx

    # Validations and cleaning
    # Ensure types
    work["fps"] = pd.to_numeric(work["fps"], errors="coerce")
    work["latency_ms"] = pd.to_numeric(work["latency_ms"], errors="coerce")
    work["cpu_percent"] = pd.to_numeric(work["cpu_percent"], errors="coerce").clip(0, 100)
    work["ram_mb"] = pd.to_numeric(work["ram_mb"], errors="coerce")
    work["frame_idx"] = pd.to_numeric(work["frame_idx"], errors="coerce").astype("Int64")

    # Derivation re-check to avoid zeros in denominators
    # If still missing one of fps/latency, derive again
    mask_missing_fps = work["fps"].isna() & work["latency_ms"].notna() & (work["latency_ms"] != 0)
    work.loc[mask_missing_fps, "fps"] = 1000.0 / work.loc[mask_missing_fps, "latency_ms"]
    mask_missing_lat = work["latency_ms"].isna() & work["fps"].notna() & (work["fps"] != 0)
    work.loc[mask_missing_lat, "latency_ms"] = 1000.0 / work.loc[mask_missing_lat, "fps"]

    # Constraints
    work = work[work["fps"] >= 0]
    work = work[work["latency_ms"] >= 0]
    work = work[work["ram_mb"].isna() | (work["ram_mb"] >= 0)]
    work["cpu_percent"] = work["cpu_percent"].clip(0, 100)

    # Final check: no NaN in standard columns
    if work[["timestamp", "fps", "latency_ms", "cpu_percent", "ram_mb", "method", "model", "scope", "file_source", "frame_idx"]].isna().any().any():
        # Drop any remaining rows with NaN to comply with "sin nulos"
        before = len(work)
        work = work.dropna(subset=STANDARD_COLUMNS)
        after = len(work)
        if after == 0:
            raise ValueError("All rows dropped due to NaN after standardization.")

    # Ensure column order and dtypes
    work = work[STANDARD_COLUMNS]
    work = work.sort_index(kind="stable")

    # Summary for this file
    summary = {
        "rows": int(len(df)),
        "raw_columns": raw_columns,
        "method": method,
        "model": model,
        "scope": scope,
        "file_source": os.path.relpath(src_path, raw_dir).replace("\\", "/"),
        "rows_kept": int(len(work)),
        "timestamp_min": pd.to_datetime(work["timestamp"]).min(),
        "timestamp_max": pd.to_datetime(work["timestamp"]).max(),
    }

    return work, summary


def process_all(raw_dir: str, out_path: str, write_readme: bool = True) -> None:
    csvs = discover_csvs(raw_dir, out_path)
    if not csvs:
        print(f"No CSV files found under {raw_dir}")
        return

    all_frames: List[pd.DataFrame] = []
    summaries: List[Dict[str, any]] = []

    for path in csvs:
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"Skipping file due to read error: {path}: {e}")
            continue

        try:
            std_df, summary = standardize_dataframe(df, path, raw_dir)
        except Exception as e:
            print(f"Skipping file due to standardization error: {path}: {e}")
            continue

        all_frames.append(std_df)
        summaries.append(summary)

        # Per-file summary line
        rel = summary["file_source"]
        scope = summary["scope"]
        print(f"[scope={scope}] {rel}: filas={summary['rows']}, columnas_crudas={summary['raw_columns']}, method/model inferidos={summary['method']}/{summary['model']}")

    if not all_frames:
        print("No standardized data produced; nothing to write.")
        return

    combined = pd.concat(all_frames, ignore_index=True)
    # Sort by timestamp then frame_idx
    combined = combined.sort_values(by=["timestamp", "frame_idx"], kind="stable")
    # Final validations
    if combined[STANDARD_COLUMNS].isna().any().any():
        # As a safety, drop rows with any NaN
        combined = combined.dropna(subset=STANDARD_COLUMNS)
    # Write output
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    combined.to_csv(out_path, index=False, encoding="utf-8")

    # Totals
    total_rows = len(combined)
    ts_min = pd.to_datetime(combined["timestamp"]).min() if total_rows else None
    ts_max = pd.to_datetime(combined["timestamp"]).max() if total_rows else None
    print(f"total_filas={total_rows}, rango_timestamp=[{ts_min}, {ts_max}]")

    # Optionally write README summary
    if write_readme:
        readme_dir = os.path.join(os.path.dirname(raw_dir), "transform")
        os.makedirs(readme_dir, exist_ok=True)
        readme_path = os.path.join(readme_dir, "README_transform.md")
        lines = []
        lines.append("# Resumen de transformación\n")
        lines.append(f"- Directorio de entrada: {raw_dir}")
        lines.append(f"- Fichero consolidado: {out_path}")
        lines.append(f"- Total de filas: {total_rows}")
        lines.append(f"- Rango de timestamp: [{ts_min}, {ts_max}]\n")
        lines.append("## Archivos procesados")
        for s in summaries:
            lines.append(
                f"- [scope={s['scope']}] {s['file_source']}: filas_crudas={s['rows']}, filas_conservadas={s['rows_kept']}, columnas_crudas={s['raw_columns']}, method/model={s['method']}/{s['model']}, rango=[{s['timestamp_min']}, {s['timestamp_max']}]"
            )
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Estandariza y consolida métricas de CSVs en un único archivo.")
    parser.add_argument(
        "--raw-dir",
        default=os.path.join("Metricas_informe", "raw"),
        help="Directorio base de entrada (por defecto: Metricas_informe/raw)",
    )
    parser.add_argument(
        "--out",
        default=os.path.join("Metricas_informe", "raw", CONSOLIDATED_BASENAME),
        help="Ruta de salida del CSV consolidado (por defecto: Metricas_informe/raw/metrica_consolidada.csv)",
    )
    parser.add_argument(
        "--no-readme",
        action="store_true",
        help="Si se especifica, omite la actualización del README_transform.md",
    )

    args = parser.parse_args()
    process_all(args.raw_dir, args.out, write_readme=not args.no_readme)


if __name__ == "__main__":
    main()