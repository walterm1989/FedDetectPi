# Resumen de transformación

Ejecuta el script transformacion.py para consolidar todas las métricas CSV bajo Metricas_informe/raw en un único archivo:

- Entrada: Metricas_informe/raw/**/*.csv (excluye metrica_consolidada.csv)
- Salida: Metricas_informe/raw/metrica_consolidada.csv
- Esquema estandarizado (orden exacto, sin nulos):
  timestamp, fps, latency_ms, cpu_percent, ram_mb, method, model, scope, file_source, frame_idx

Sinónimos de columnas aceptadas:
- timestamp: ["timestamp","time","datetime","frame_time","ts"]
- fps: ["fps","FPS","frames_per_second","fps_inst"]
- latency_ms: ["latency_ms","latency","lat_ms","inference_ms","inference_time_ms","latencia_ms"]
- cpu_percent: ["cpu_percent","cpu","CPU_%","cpu_usage","cpu_use","cpu_pct"]
- ram_mb: ["ram_mb","memory_mb","mem_mb","ram_usage_mb","rss_mb","memory"]
- frame_idx: ["frame","frame_idx","index"]
- method cruda (raw_method): ["method"]

Inferencias:
- scope: "flower" si la ruta contiene /FlowerAI/, "local" si contiene /Local/
- method/model: a partir de raw_method o nombre de archivo
  - method: "BoundingBoxes" si contiene "BBoxes", "KeyPoints" si contiene "KeyPoints"
  - model: intenta detectar YOLOv*, ResNet50, MobileNetV2, ViT; si no, "unknown"

Notas:
- No se leen ficheros de salida generados por el propio proceso (metrica_consolidada.csv)
- Muestra un resumen por consola (filas por archivo, carpetas, columnas, total y rango de timestamps)