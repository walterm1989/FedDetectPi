import csv
import glob
import os
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Estandar final requerido (orden exacto)
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

# Sinónimos de columnas de entrada
SYNONYMS: Dict[str, List[str]] = {
    "timestamp": ["timestamp", "time", "datetime", "frame_time", "ts"],
    "fps": ["fps", "FPS", "frames_per_second", "fps_inst"],
    "latency_ms": ["latency_ms", "latency", "lat_ms", "inference_ms", "inference_time_ms", "latencia_ms"],
    "cpu_percent": ["cpu_percent", "cpu", "CPU_%", "cpu_usage", "cpu_use", "cpu_pct"],
    "ram_mb": ["ram_mb", "memory_mb", "mem_mb", "ram_usage_mb", "rss_mb", "memory"],
    "frame_idx": ["frame", "frame_idx", "index"],
    # "method" especial: tratar como raw_method si existe
    "raw_method": ["method"],
}

OUTPUT_FILE = os.path.join("Metricas_informe", "raw", "metrica_consolidada.csv")
RAW_GLOB = os.path.join("Metricas_informe", "raw", "**", "*.csv")

# Utilidades de parseo/normalización


def normalized_decimal(value: str) -> Optional[float]:
    """
    Convierte cadenas con comas, % y unidades varias a float.
    Devuelve None si no se puede convertir.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    # eliminar símbolos comunes
    s = s.replace("%", "")
    # detectar unidades de memoria
    s_bytes = False
    # 123 MiB / 123 MB / 123.4MB
    mib_match = re.search(r"([0-9]+(?:[\\.,][0-9]+)?)\\s*(mib|mb|gb|gib|kb|kib|b)\\b", s, flags=re.I)
    if mib_match:
        num_str, unit = mib_match.group(1), mib_match.group(2).lower()
        num_str = num_str.replace(",", ".")
        try:
            v = float(num_str)
        except ValueError:
            return None
        # Convertir a bytes para manejo consistente y marcar
        if unit in {"b"}:
            v_bytes = v
        elif unit in {"kb", "kib"}:
            v_bytes = v * 1024
        elif unit in {"mb", "mib"}:
            v_bytes = v * 1024 * 1024
        elif unit in {"gb", "gib"}:
            v_bytes = v * 1024 * 1024 * 1024
        else:
            v_bytes = v
        s_bytes = True
        return float(v_bytes)

    # reemplazar coma decimal por punto
    s = s.replace(",", ".")
    # eliminar espacios
    s = re.sub(r"\\s+", "", s)
    # eliminar sufijos alfanuméricos residuales
    s = re.sub(r"[^0-9\\.\\-eE+]", "", s)

    try:
        v = float(s)
        return v
    except ValueError:
        return None


def to_ram_mb(value: str) -> Optional[float]:
    """
    Convierte posibles representaciones de memoria a MB.
    Heurística:
      - Si se detectó en bytes (muy grande), convertir a MB.
      - Si ya parece MB (rango típico 50..64000), dejar como está.
      - Si parece GB (0.1..1024), convertir a MB si detectamos sufijos.
    """
    v = normalized_decimal(value)
    if v is None:
        return None
    # Heurística: si es muy grande, interpretamos como bytes y convertimos a MB
    if v > 1024 * 1024 * 16:  # mayor que 16MB en bytes
        return v / (1024.0 * 1024.0)
    # Si está en un rango razonable de MB, devolver tal cual
    return float(v)


def parse_timestamp(value: str) -> Tuple[str, Optional[float]]:
    """
    Intenta normalizar timestamp a ISO-8601. Devuelve (iso_string, epoch_seconds or None).
    Si no puede convertir, devuelve una constante por defecto y None para epoch.
    """
    if value is None:
        return "1970-01-01T00:00:00Z", None
    s = str(value).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return "1970-01-01T00:00:00Z", None

    # Si parece epoch (segundos o milisegundos)
    if re.fullmatch(r"[0-9]{10}", s):  # seconds
        ts = int(s)
        iso = datetime.utcfromtimestamp(ts).isoformat() + "Z"
        return iso, float(ts)
    if re.fullmatch(r"[0-9]{13}", s):  # milliseconds
        ts = int(s) / 1000.0
        iso = datetime.utcfromtimestamp(ts).isoformat() + "Z"
        return iso, float(ts)

    # Intentar parseo flexible de ISO/datetime comunes
    fmts = [
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
    ]
    for fmt in fmts:
        try:
            dt = datetime.strptime(s, fmt)
            iso = dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
            return iso, dt.timestamp()
        except Exception:
            pass
    # Último recurso: devolver tal cual si parece ISO
    if re.match(r"\\d{4}-\\d{2}-\\d{2}T", s):
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            iso = dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
            return iso, dt.timestamp()
        except Exception:
            return s, None

    return "1970-01-01T00:00:00Z", None


def detect_scope(path: str) -> str:
    p = path.replace("\\\\", "/")
    if "/FlowerAI/" in p:
        return "flower"
    if "/Local/" in p:
        return "local"
    return "unknown"


def infer_method_model_from_raw(raw: Optional[str], file_name: str) -> Tuple[str, str]:
    """
    A partir de una cadena tipo 'BBoxes-YOLOv4tiny' o 'KeyPoints-resnet50' extrae:
    method: 'BoundingBoxes' o 'KeyPoints'
    model: 'YOLOv4tiny', 'ResNet50', etc. si detectable
    Si no hay raw_method, intenta inferir por nombre de archivo.
    """
    source = (raw or "") + " " + (file_name or "")
    s = source.lower()

    # method
    if "bboxes" in s or "bbox" in s or "boundingboxes" in s:
        method = "BoundingBoxes"
    elif "keypoints" in s or "keypoint" in s or "kp" in s:
        method = "KeyPoints"
    else:
        method = "unknown"

    # model
    model = "unknown"
    # YOLO family
    m = re.search(r"(yolov\\d+t?iny|yolov\\d+|yolo[v\\-]?\\d+\\w*)", s)
    if m:
        model = m.group(1)
    elif "resnet-50" in s or "resnet50" in s or re.search(r"resnet\\s*50", s):
        model = "ResNet50"
    elif "mobilenetv2" in s or "mobilenet-v2" in s:
        model = "MobileNetV2"
    elif "vit" in s:
        model = "ViT"
    # Normalizaciones puntuales
    if model.lower() in {"resnet-50", "resnet50"}:
        model = "ResNet50"
    return method, model


def resolve_header_mapping(headers: List[str]) -> Dict[str, str]:
    """
    Devuelve un mapping standard_col -> input_header_name si existe.
    Manejo especial: 'method' de entrada se considera 'raw_method' y se evita usar directamente.
    """
    lower_map = {h.lower(): h for h in headers}
    mapping: Dict[str, str] = {}

    for std_col, syns in SYNONYMS.items():
        for alias in syns:
            if alias.lower() in lower_map:
                mapping[std_col] = lower_map[alias.lower()]
                break
    # Evitar mapear method final por accidente; solo raw_method
    if "method" in mapping:
        # si por accidente se metió 'method' directo, lo movemos a raw_method
        mapping["raw_method"] = mapping.pop("method")
    return mapping


def consolidate():
    files = glob.glob(RAW_GLOB, recursive=True)
    files = [f for f in files if os.path.basename(f) != os.path.basename(OUTPUT_FILE)]
    files = [f for f in files if os.path.isfile(f)]

    print("Descubriendo CSVs...")
    for f in files:
        print(f" - {f}")
    if not files:
        print("No se encontraron CSVs de entrada.")
        return

    consolidated_rows: List[List[str]] = []
    per_file_counts: List[Tuple[str, int]] = []
    per_file_headers: List[Tuple[str, List[str]]] = []
    global_min_ts: Optional[float] = None
    global_max_ts: Optional[float] = None
    total_rows = 0

    for path in files:
        scope = detect_scope(path)
        file_source = os.path.splitext(os.path.basename(path))[0]

        try:
            with open(path, "r", newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                headers = reader.fieldnames or []
                per_file_headers.append((path, headers))
                mapping = resolve_header_mapping(headers)

                # Columna cpu_percent: si parece fraccional en todo el archivo, multiplicar por 100
                # Necesitamos mirar el primer bloque para estimar
                rows_buffer = list(reader)
                cpu_vals: List[float] = []
                mem_vals: List[float] = []

                for r in rows_buffer[:200]:
                    # cpu
                    src_cpu = r.get(mapping.get("cpu_percent", ""), None) if "cpu_percent" in mapping else None
                    vcpu = normalized_decimal(src_cpu) if src_cpu is not None else None
                    if vcpu is not None:
                        cpu_vals.append(vcpu)
                    # memory
                    src_mem = r.get(mapping.get("ram_mb", ""), None) if "ram_mb" in mapping else None
                    vmem = normalized_decimal(src_mem) if src_mem is not None else None
                    if vmem is not None:
                        mem_vals.append(vmem)

                cpu_scale_100 = False
                if cpu_vals:
                    mx = max(cpu_vals)
                    # heurística: si todos <= 1.5 asumimos 0..1
                    if mx <= 1.5:
                        cpu_scale_100 = True

                # memory heurística: si parecen bytes muy altos en general
                mem_bytes_assume = False
                if mem_vals:
                    if max(mem_vals) > 1024 * 1024 * 16:  # >16MB en bytes
                        mem_bytes_assume = True

                # Procesar filas
                file_row_count = 0
                # Si existe raw_method, lo usaremos para inferir finales
                raw_method_col = mapping.get("raw_method")
                for row in rows_buffer:
                    out: Dict[str, str] = {}

                    # timestamp
                    ts_val = row.get(mapping.get("timestamp", ""), None) if "timestamp" in mapping else None
                    ts_iso, ts_epoch = parse_timestamp(ts_val)
                    out["timestamp"] = ts_iso
                    if ts_epoch is not None:
                        global_min_ts = ts_epoch if global_min_ts is None else min(global_min_ts, ts_epoch)
                        global_max_ts = ts_epoch if global_max_ts is None else max(global_max_ts, ts_epoch)

                    # fps
                    fps_val = row.get(mapping.get("fps", ""), None) if "fps" in mapping else None
                    fps = normalized_decimal(fps_val)
                    out["fps"] = f"{0.0 if fps is None else float(fps):.3f}"

                    # latency_ms
                    lat_val = row.get(mapping.get("latency_ms", ""), None) if "latency_ms" in mapping else None
                    lat = normalized_decimal(lat_val)
                    out["latency_ms"] = f"{0.0 if lat is None else float(lat):.3f}"

                    # cpu_percent
                    cpu_val = row.get(mapping.get("cpu_percent", ""), None) if "cpu_percent" in mapping else None
                    cpu = normalized_decimal(cpu_val)
                    if cpu is None:
                        cpu = 0.0
                    if cpu_scale_100:
                        cpu = cpu * 100.0
                    out["cpu_percent"] = f"{float(cpu):.3f}"

                    # ram_mb
                    mem_val = row.get(mapping.get("ram_mb", ""), None) if "ram_mb" in mapping else None
                    mem = normalized_decimal(mem_val)
                    if mem is None:
                        mem_mb = 0.0
                    else:
                        if mem_bytes_assume or (mem is not None and mem > 1024 * 1024 * 16):
                            mem_mb = mem / (1024.0 * 1024.0)
                        else:
                            # adicional, si trae unidad en texto ya lo maneja to_ram_mb
                            mem_mb = to_ram_mb(str(mem_val)) if mem_val is not None else float(mem)
                            if mem_mb is None:
                                mem_mb = float(mem)
                    out["ram_mb"] = f"{float(mem_mb):.3f}"

                    # file_source
                    out["file_source"] = file_source

                    # scope
                    out["scope"] = scope

                    # frame_idx
                    fidx_val = row.get(mapping.get("frame_idx", ""), None) if "frame_idx" in mapping else None
                    fidx = normalized_decimal(fidx_val)
                    out["frame_idx"] = str(int(fidx)) if fidx is not None else "-1"

                    # method/model: a partir de raw_method o nombre archivo
                    raw_method_value = row.get(raw_method_col, None) if raw_method_col else None
                    method, model = infer_method_model_from_raw(raw_method_value, os.path.basename(path))
                    out["method"] = method
                    out["model"] = model if model else "unknown"

                    # Asegurar no nulos y orden correcto
                    consolidated_rows.append([out.get(col, "unknown") for col in STANDARD_COLUMNS])
                    file_row_count += 1

                per_file_counts.append((path, file_row_count))
                total_rows += file_row_count

        except Exception as e:
            print(f"Error leyendo {path}: {e}")

    # Escribir consolidado
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(STANDARD_COLUMNS)
        writer.writerows(consolidated_rows)

    # Resumen
    print("\nResumen de consolidación")
    print("------------------------")
    print(f"Archivos procesados: {len(per_file_counts)}")
    for path, cnt in per_file_counts:
        folder = "FlowerAI" if "/FlowerAI/" in path.replace("\\\\", "/") else ("Local" if "/Local/" in path.replace("\\\\", "/") else "Otro")
        cols = next((h for p, h in per_file_headers if p == path), [])
        print(f"- {path} | carpeta={folder} | filas={cnt} | columnas={len(cols)} -> {cols}")
    print(f"Total de filas: {total_rows}")
    if global_min_ts is not None and global_max_ts is not None:
        rmin = datetime.utcfromtimestamp(global_min_ts).isoformat() + "Z"
        rmax = datetime.utcfromtimestamp(global_max_ts).isoformat() + "Z"
        print(f"Rango de timestamps (UTC): {rmin} .. {rmax}")
    else:
        print("Rango de timestamps (UTC): no disponible")

    # (Opcional) Actualizar/crear README_transform.md con el resumen
    try:
        readme_path = os.path.join("Metricas_informe", "transform", "README_transform.md")
        lines: List[str] = []
        lines.append("# Resumen de transformación\n")
        lines.append(f"- Fecha de ejecución: {datetime.utcnow().isoformat()}Z\n")
        lines.append(f"- Archivos procesados: {len(per_file_counts)}\n")
        for path, cnt in per_file_counts:
            folder = "FlowerAI" if "/FlowerAI/" in path.replace("\\\\", "/") else ("Local" if "/Local/" in path.replace("\\\\", "/") else "Otro")
            cols = next((h for p, h in per_file_headers if p == path), [])
            lines.append(f"  - {path} | carpeta={folder} | filas={cnt} | columnas={len(cols)} -> {cols}\n")
        lines.append(f"- Total de filas: {total_rows}\n")
        if global_min_ts is not None and global_max_ts is not None:
            rmin = datetime.utcfromtimestamp(global_min_ts).isoformat() + "Z"
            rmax = datetime.utcfromtimestamp(global_max_ts).isoformat() + "Z"
            lines.append(f"- Rango de timestamps (UTC): {rmin} .. {rmax}\n")
        else:
            lines.append("- Rango de timestamps (UTC): no disponible\n")
        os.makedirs(os.path.dirname(readme_path), exist_ok=True)
        with open(readme_path, "w", encoding="utf-8") as rh:
            rh.writelines(lines)
    except Exception as e:
        print(f"No se pudo actualizar README_transform.md: {e}")


if __name__ == "__main__":
    consolidate()