#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformación de métricas (local + Flower AI) a un esquema estándar.

- Lee exactamente 4 CSV desde Metricas_informe/raw (o --raw-dir)
- Estandariza columnas, tipos y valores
- Infere method/model/scope por nombre de archivo
- Genera metrica_consolidada.csv en Metricas_informe/transform/ (o --out)
- Imprime un resumen por consola
- (Opcional) Escribe README_transform.md con el resumen de mapeos/validaciones

Requisitos: Python 3.9+, pandas
"""

from __future__ import annotations

import argparse
import sys
import re
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# Esquema final y orden de columnas
FINAL_COLUMNS = [
    "timestamp",     # string ISO8601
    "fps",           # float
    "latency_ms",    # float
    "cpu_percent",   # float 0..100
    "ram_mb",        # float
    "method",        # str: BoundingBoxes | KeyPoints
    "model",         # str: YOLOv4tiny | ResNet50 | YOLO | unknown
    "scope",         # str: local | flower
    "file_source",   # str: nombre de archivo original
    "frame_idx",     # int
]

# Sinónimos -> columna estándar (case-sensitive en crudo; se eliminarán espacios)
SYNONYMS: Dict[str, List[str]] = {
    "timestamp": ["timestamp", "time", "datetime", "frame_time", "ts"],
    "fps": ["fps", "FPS", "frames_per_second"],
    "latency_ms": ["latency_ms", "latency", "lat_ms", "inference_ms", "inference_time_ms"],
    "cpu_percent": ["cpu_percent", "cpu", "CPU_%", "cpu_usage", "cpu_use"],
    "ram_mb": ["ram_mb", "memory_mb", "mem_mb", "ram_usage_mb", "rss_mb", "memory"],
    "frame_idx": ["frame", "frame_idx", "index"],
}

# Archivos esperados (deben existir)
EXPECTED_FILES = [
    "metricas_boundingboxes.csv",
    "metricas_keypoints.csv",
    "20250909_194459_BBoxes-YOLOv4tiny_webcam.csv",
    "20250911_210416_KeyPoints-resnet50_webcam.csv",
]

# Columnas clave que deben existir tras mapeo
REQUIRED_METRICS = ["fps", "latency_ms", "cpu_percent", "ram_mb"]


@dataclass
class FileSummary:
    file_name: str
    rows_before: int
    rows_after: int
    detected_columns: Dict[str, str]  # estandar -> crudo
    warnings: List[str]


def _strip_and_collapse_spaces(cols: List[str]) -> List[str]:
    """Elimina espacios alrededor y dentro (colapsa) de los nombres de columnas."""
    cleaned = []
    for c in cols:
        if c is None:
            cleaned.append("")
            continue
        # Eliminar espacios en exceso, tabs, etc.
        cc = re.sub(r"\s+", "", str(c))
        cleaned.append(cc)
    return cleaned


def _find_mapping(columns: List[str]) -> Tuple[Dict[str, str], List[str]]:
    """
    Determina el mapeo de columnas crudas -> estándar usando los sinónimos.
    Devuelve:
      - mapping: estándar -> crudo
      - warnings: lista de advertencias
    """
    warnings = []
    mapping: Dict[str, str] = {}

    for standard, alias_list in SYNONYMS.items():
        found_raw: Optional[str] = None
        # Buscar primera coincidencia en orden de preferencia
        for alias in alias_list:
            for raw in columns:
                if raw == alias:
                    found_raw = raw
                    break
            if found_raw is not None:
                break
        if found_raw:
            mapping[standard] = found_raw

    # Avisar si hay múltiples candidatos (muy raro por igualdad exacta)
    # También capturar columnas inesperadas
    known_raw = set(mapping.values())
    unexpected = [c for c in columns if c not in known_raw]
    if unexpected:
        warnings.append(f"Columnas no mapeadas (ignoradas): {unexpected}")

    return mapping, warnings


def _infer_method_model_scope(file_name: str) -> Tuple[str, str, str]:
    """Infere method, model y scope desde el nombre de archivo (case-insensitive)."""
    lower = file_name.lower()

    # method
    if ("bboxes" in lower) or ("bounding" in lower):
        method = "BoundingBoxes"
    elif ("keypoints" in lower) or ("keypoint" in lower):
        method = "KeyPoints"
    else:
        method = "unknown"

    # model
    model = "unknown"
    # YOLO detection (extraer subcadena YOLO*)
    yolo_match = re.search(r"(yolo[a-z0-9\-]*)", lower, flags=re.IGNORECASE)
    if yolo_match:
        # Preservar capitalización típica (YOLOv4tiny)
        candidate = yolo_match.group(1)
        # Normalizar primeras letras
        # Dejar YOLO en mayúsculas, resto como en original
        model = "YOLO" + candidate[4:]
        if model == "YOLO":
            model = "YOLO"
    elif "resnet50" in lower or "resnet-50" in lower:
        model = "ResNet50"

    # scope
    if file_name in ("metricas_boundingboxes.csv", "metricas_keypoints.csv"):
        scope = "local"
    elif file_name in (
        "20250909_194459_BBoxes-YOLOv4tiny_webcam.csv",
        "20250911_210416_KeyPoints-resnet50_webcam.csv",
    ):
        scope = "flower"
    else:
        # Heurística adicional
        if "webcam" in lower or "flower" in lower:
            scope = "flower"
        else:
            scope = "local"

    return method, model, scope


def _parse_possible_timestamp_series(s: pd.Series) -> pd.Series:
    """
    Convierte una serie de timestamps a ISO8601.
    - Si es numérica, intenta epoch segundos y luego milisegundos.
    - Si es string, usa pandas.to_datetime para parsear.
    Devuelve series de str ISO8601 (UTC).
    """
    if pd.api.types.is_numeric_dtype(s):
        # Heurística: detectar escala
        s_nonnull = s.dropna().astype(float)
        if s_nonnull.empty:
            dt = pd.to_datetime(s, unit="s", utc=True, errors="coerce")
        else:
            mx = s_nonnull.max()
            # Si valores grandes, probablemente ms
            if mx > 1e11:
                dt = pd.to_datetime(s, unit="ms", utc=True, errors="coerce")
            else:
                dt = pd.to_datetime(s, unit="s", utc=True, errors="coerce")
    else:
        dt = pd.to_datetime(s, utc=True, errors="coerce")

    # A ISO8601 con 'Z'
    return dt.dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _clean_numeric(series: pd.Series, is_percent: bool = False) -> pd.Series:
    """Convierte una serie a float, manejando comas decimales y símbolos %."""
    if series is None:
        return series
    s = series.astype(str).str.strip()
    if is_percent:
        s = s.str.replace("%", "", regex=False)
    # Cambiar coma decimal por punto
    s = s.str.replace(",", ".", regex=False)
    # Vacíos a NaN
    s = s.replace({"": pd.NA, "None": pd.NA, "nan": pd.NA})
    return pd.to_numeric(s, errors="coerce").astype(float)


def _ensure_cpu_percent_scale(s: pd.Series) -> pd.Series:
    """
    Asegura que cpu_percent está en 0..100.
    Si parece escala 0..1, multiplica por 100.
    """
    if s.dropna().empty:
        return s
    mx = s.max()
    if mx is not None and pd.notna(mx) and mx <= 1.5:
        return s * 100.0
    return s


def _generate_timestamps_from_index(
    n: int,
    fps: Optional[float],
    file_name: str,
    start_dt: Optional[datetime] = None,
) -> List[str]:
    """
    Genera n timestamps ISO8601 a partir de fps y un inicio.
    - Intenta inferir inicio desde prefijo YYYYMMDD_HHMMSS en file_name.
    - Si fps es None, usa 5 Hz (0.2 s por frame).
    """
    # Intentar parsear prefijo en nombre de archivo
    if start_dt is None:
        m = re.match(r"(\d{8})_(\d{6})", file_name)
        if m:
            date_part = m.group(1)
            time_part = m.group(2)
            try:
                start_dt = datetime.strptime(f"{date_part}{time_part}", "%Y%m%d%H%M%S")
                start_dt = start_dt.replace(tzinfo=timezone.utc)
            except Exception:
                start_dt = None

    if start_dt is None:
        start_dt = datetime.now(timezone.utc)

    if fps is None or not pd.notna(fps) or fps <= 0:
        delta = timedelta(seconds=0.2)  # fallback
    else:
        delta = timedelta(seconds=1.0 / float(fps))

    stamps = []
    t = start_dt
    for _ in range(n):
        stamps.append(t.strftime("%Y-%m-%dT%H:%M:%S.%fZ"))
        t = t + delta
    return stamps


def _standardize_single_csv(path: Path) -> Tuple[pd.DataFrame, FileSummary]:
    """Lee y estandariza un CSV individual según el esquema final."""
    df_raw = pd.read_csv(path)
    rows_before = len(df_raw)

    # Limpiar nombres de columnas crudos (eliminar espacios)
    df = df_raw.copy()
    df.columns = _strip_and_collapse_spaces(list(df.columns))
    mapping, warnings = _find_mapping(list(df.columns))

    detected_columns = {std: mapping[std] for std in mapping.keys()}

    # Validar presencia de métricas clave
    missing_required = [col for col in REQUIRED_METRICS if col not in mapping]
    if missing_required:
        raise ValueError(
            f"Archivo '{path.name}' no contiene métricas requeridas: {missing_required}. "
            f"Columnas detectadas: {list(df.columns)}"
        )

    # Construir DataFrame estándar con columnas mapeadas
    out = pd.DataFrame()
    # fps
    out["fps"] = _clean_numeric(df[mapping["fps"]])
    # latency_ms
    out["latency_ms"] = _clean_numeric(df[mapping["latency_ms"]])
    # cpu_percent
    out["cpu_percent"] = _clean_numeric(df[mapping["cpu_percent"]], is_percent=True)
    out["cpu_percent"] = _ensure_cpu_percent_scale(out["cpu_percent"])
    # ram_mb
    out["ram_mb"] = _clean_numeric(df[mapping["ram_mb"]])

    # frame_idx (opcional)
    if "frame_idx" in mapping:
        # Algunos índices pueden venir como float por comas: forzar int tras limpieza
        fi_clean = _clean_numeric(df[mapping["frame_idx"]])
        # Rellenar posibles NaN con rango
        if fi_clean.isna().any():
            fi_clean = fi_clean.fillna(pd.Series(range(len(df)), index=df.index)).astype(float)
        out["frame_idx"] = fi_clean.astype(int)
    else:
        out["frame_idx"] = pd.Series(range(len(df)), index=df.index, dtype=int)

    # timestamp (opcional)
    if "timestamp" in mapping:
        ts_series = _parse_possible_timestamp_series(df[mapping["timestamp"]])
        # Si hubo NaT -> completar con serie sintética
        if ts_series.isna().any() or (ts_series == "NaT").any():
            # Determinar fps promedio razonable
            fps_val = float(out["fps"].dropna().median()) if not out["fps"].dropna().empty else None
            gen = _generate_timestamps_from_index(len(df), fps=fps_val, file_name=path.name)
            ts_series = pd.Series(gen, index=df.index)
        out["timestamp"] = ts_series
    else:
        fps_val = float(out["fps"].dropna().median()) if not out["fps"].dropna().empty else None
        gen = _generate_timestamps_from_index(len(df), fps=fps_val, file_name=path.name)
        out["timestamp"] = pd.Series(gen, index=df.index)

    # method/model/scope/file_source por nombre de archivo
    method, model, scope = _infer_method_model_scope(path.name)
    out["method"] = method
    out["model"] = model
    out["scope"] = scope
    out["file_source"] = path.name

    # Validaciones y normalizaciones finales por archivo
    # Valores no negativos
    for col in ["fps", "latency_ms", "ram_mb"]:
        if out[col].lt(0).any():
            warnings.append(f"Se detectaron valores negativos en {col}; se forzarán a NaN.")
            out.loc[out[col] &lt; 0, col] = pd.NA

    # cpu_percent debe estar en 0..100
    if out["cpu_percent"].lt(0).any() or out["cpu_percent"].gt(100).any():
        # Intentar corregir escalas extrañas: si max &lt;= 1.5 tras limpiar, ya se multiplicó
        # Si aún hay valores &gt; 100, limitar a 100 (warning)
        too_big = out["cpu_percent"].gt(100)
        if too_big.any():
            warnings.append("cpu_percent &gt; 100 detectado; se limitará a 100.")
            out.loc[too_big, "cpu_percent"] = 100.0
        too_low = out["cpu_percent"].lt(0)
        if too_low.any():
            warnings.append("cpu_percent &lt; 0 detectado; se forzará a 0.")
            out.loc[too_low, "cpu_percent"] = 0.0

    # Casteos finales
    out["fps"] = out["fps"].astype(float)
    out["latency_ms"] = out["latency_ms"].astype(float)
    out["cpu_percent"] = out["cpu_percent"].astype(float)
    out["ram_mb"] = out["ram_mb"].astype(float)
    out["frame_idx"] = out["frame_idx"].astype(int)
    out["timestamp"] = out["timestamp"].astype(str)

    rows_after = len(out)
    summary = FileSummary(
        file_name=path.name,
        rows_before=rows_before,
        rows_after=rows_after,
        detected_columns=detected_columns,
        warnings=warnings,
    )
    return out, summary


def _validate_final(df: pd.DataFrame):
    """Valida el DataFrame final según las reglas."""
    # Columnas presentes y orden
    missing_cols = [c for c in FINAL_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Faltan columnas en consolidado: {missing_cols}")

    # Sin nulos
    if df[FINAL_COLUMNS].isna().any().any():
        # Localizar primeras ocurrencias de nulos
        null_counts = df[FINAL_COLUMNS].isna().sum()
        raise ValueError(f"Se detectaron valores nulos tras consolidación: {null_counts.to_dict()}")

    # Rango de métricas
    if (df["fps"] &lt; 0).any():
        raise ValueError("fps contiene valores negativos.")
    if (df["latency_ms"] &lt; 0).any():
        raise ValueError("latency_ms contiene valores negativos.")
    if (df["ram_mb"] &lt; 0).any():
        raise ValueError("ram_mb contiene valores negativos.")
    if (df["cpu_percent"] &lt; 0).any() or (df["cpu_percent"] &gt; 100).any():
        raise ValueError("cpu_percent fuera de rango [0,100].")

    # Al menos 4 archivos leídos y filas &gt; 0 se verifica fuera (con summaries)


def _print_console_summary(summaries: List[FileSummary], final_df: pd.DataFrame):
    """Imprime resumen legible por consola."""
    print("Resumen de transformación")
    print("=========================")
    for s in summaries:
        print(f"- Archivo: {s.file_name}")
        print(f"  Filas (antes): {s.rows_before}")
        print(f"  Filas (después): {s.rows_after}")
        print(f"  Columnas detectadas (estándar -> crudo):")
        for std, raw in s.detected_columns.items():
            print(f"    {std}: {raw}")
        if s.warnings:
            print(f"  Warnings:")
            for w in s.warnings:
                print(f"    - {w}")
        print("")

    print("Totales")
    print("-------")
    print(f"Archivos procesados: {len(summaries)}")
    print(f"Filas totales finales: {len(final_df)}")
    if not final_df.empty:
        ts_min = final_df['timestamp'].min()
        ts_max = final_df['timestamp'].max()
        print(f"Rango de timestamps: {ts_min} .. {ts_max}")
    print("")


def _write_readme_transform(out_dir: Path, summaries: List[FileSummary], final_df: pd.DataFrame):
    """Escribe un README opcional con el resumen de mapeos y validaciones."""
    lines = []
    lines.append("# Transformación de métricas")
    lines.append("")
    lines.append("Este documento resume los mapeos y validaciones aplicados por transformacion.py.")
    lines.append("")
    lines.append("## Archivos procesados")
    for s in summaries:
        lines.append(f"- {s.file_name}: {s.rows_before} filas -> {s.rows_after} filas")
    lines.append("")
    lines.append("## Mapeos detectados (estándar -> crudo)")
    for s in summaries:
        lines.append(f"### {s.file_name}")
        for std, raw in s.detected_columns.items():
            lines.append(f"- {std} -> {raw}")
        if s.warnings:
            lines.append(f"- Warnings:")
            for w in s.warnings:
                lines.append(f"  - {w}")
        lines.append("")
    lines.append("## Esquema final y validaciones")
    lines.append(f"- Columnas finales: {', '.join(FINAL_COLUMNS)}")
    lines.append("- Reglas: sin nulos; fps, latency_ms, ram_mb >= 0; cpu_percent en [0, 100].")
    if not final_df.empty:
        ts_min = final_df['timestamp'].min()
        ts_max = final_df['timestamp'].max()
        lines.append(f"- Rango de timestamps: {ts_min} .. {ts_max}")
        lines.append(f"- Filas totales: {len(final_df)}")
    lines.append("")

    readme_path = out_dir / "README_transform.md"
    readme_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Estandariza y consolida métricas de local y Flower AI.")
    parser.add_argument("--raw-dir", default="Metricas_informe/raw", help="Directorio de entrada con los CSV crudos")
    parser.add_argument(
        "--out",
        default="Metricas_informe/transform/metrica_consolidada.csv",
        help="Ruta de salida del CSV consolidado",
    )
    parser.add_argument(
        "--no-readme",
        action="store_true",
        help="No generar README_transform.md con el resumen",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_path = Path(args.out)
    out_dir = out_path.parent

    # Crear carpeta de salida si no existe
    out_dir.mkdir(parents=True, exist_ok=True)

    # Validar presencia de los 4 archivos esperados
    missing = []
    paths: List[Path] = []
    for fname in EXPECTED_FILES:
        p = raw_dir / fname
        if not p.exists():
            missing.append(fname)
        else:
            paths.append(p)

    if missing:
        raise FileNotFoundError(
            f"Faltan archivos esperados en '{raw_dir}': {missing}. "
            f"Se esperaban exactamente: {EXPECTED_FILES}"
        )

    # Procesar cada archivo
    standardized: List[pd.DataFrame] = []
    summaries: List[FileSummary] = []
    for p in paths:
        try:
            std_df, summary = _standardize_single_csv(p)
        except Exception as e:
            print(f"Error procesando '{p.name}': {e}", file=sys.stderr)
            raise
        standardized.append(std_df)
        summaries.append(summary)

    # Concatenar y ordenar
    consolidated = pd.concat(standardized, ignore_index=True)

    # Ordenar por timestamp y luego frame_idx
    consolidated = consolidated.sort_values(by=["timestamp", "frame_idx"], kind="stable").reset_index(drop=True)

    # Reordenar columnas al esquema final
    consolidated = consolidated[FINAL_COLUMNS]

    # Validación final
    _validate_final(consolidated)

    # Guardar CSV
    consolidated.to_csv(out_path, index=False, encoding="utf-8")

    # Resumen por consola
    _print_console_summary(summaries, consolidated)

    # README opcional
    if not args.no_readme:
        _write_readme_transform(out_dir, summaries, consolidated)


if __name__ == "__main__":
    main()