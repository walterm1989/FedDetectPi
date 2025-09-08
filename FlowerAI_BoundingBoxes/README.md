# FlowerAI_BoundingBoxes

Detección de personas en tiempo real con OpenCV HOG+SVM (Bounding Boxes) y plano de control con Flower AI (sin entrenamiento). Diseñado para Raspberry Pi 4 (64-bit) y portátil.

Nota: Flower se usa exclusivamente como plano de control para configurar parámetros de detección (threshold, win_stride, padding, scale). El método de detección es siempre HOG+SVM y en los CSV se reporta como BBoxes-HOG.

## Estructura

- server.py: Servidor Flower (FedAvg, sin entrenamiento) que envía config a los clientes.
- client_raspberry.py: Cliente que realiza la detección con HOG+SVM y reporta métricas a CSV. Lee la configuración del servidor en cada ronda de Flower y actualiza parámetros al vuelo. Si el servidor no está disponible, continúa en modo local.
- utils/metrics.py: Utilidad para crear y escribir el CSV con cabecera exacta y flush seguro.
- utils/camera.py: Utilidad para abrir la cámara/webcam probando índices, y soporte para vídeo desde archivo.
- requirements.txt: Dependencias (versiones compatibles con Raspberry Pi 64-bit).
- run_server.sh / run_client.sh: Scripts opcionales con ejemplos de ejecución.
- Metrics/: Directorio de salida por defecto para CSV (se crea automáticamente si no existe).

## Requisitos

- Python 3.9 o superior.
- OpenCV 4.x.
- En Raspberry Pi, preferir opencv-python-headless si no se usa ventana de visualización (--show=False).

Instalar dependencias (Raspberry/portátil):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r FlowerAI_BoundingBoxes/requirements.txt
```

En Raspberry Pi 64-bit, opencv-python-headless evita dependencias de GUI. Si se necesita mostrar ventana (--show=True), instalar `opencv-python` en lugar de `opencv-python-headless`.

## Ejecución

### Servidor Flower (plano de control)

Ejemplo (0.0.0.0:8080, 9999 rondas, threshold 0.5, win_stride 8, padding 8, scale 1.05):

```bash
python FlowerAI_BoundingBoxes/server.py \
  --address 0.0.0.0:8080 \
  --rounds 9999 \
  --threshold 0.5 \
  --win-stride 8 \
  --padding 8 \
  --scale 1.05
```

### Cliente (Raspberry/portátil) con webcam

```bash
python FlowerAI_BoundingBoxes/client_raspberry.py \
  --server 127.0.0.1:8080 \
  --source webcam --cam-index 0 \
  --duration 60 \
  --out-dir ./FlowerAI_BoundingBoxes/Metrics
```

### Cliente con archivo de vídeo

```bash
python FlowerAI_BoundingBoxes/client_raspberry.py \
  --server 192.168.1.10:8080 \
  --source video --video-path ./samples/people.mp4 \
  --duration 60 \
  --out-dir ./FlowerAI_BoundingBoxes/Metrics
```

## Parámetros del cliente

- --server: host:puerto del servidor Flower. Si no está disponible, el cliente continúa con valores por defecto.
- --source: webcam|video
- --cam-index: índice de cámara (si --source=webcam). Si falla, se prueban los índices 0..3 automáticamente.
- --video-path: ruta del archivo de vídeo (si --source=video).
- --duration: duración en segundos (por defecto 60).
- --show: muestra ventana con bounding boxes (False por defecto).
- --out-dir: directorio de salida para CSV (por defecto ./FlowerAI_BoundingBoxes/Metrics).
- --threshold, --win-stride, --padding, --scale: valores por defecto locales; pueden ser sobreescritos por el servidor en tiempo de ejecución si hay conexión.

## CSV de salida

Se genera en `FlowerAI_BoundingBoxes/Metrics/` con nombre:

- webcam: `YYYYMMDD_HHMMSS_BBoxes-HOG_webcam.csv`
- vídeo: `YYYYMMDD_HHMMSS_BBoxes-HOG_video-&lt;nombre&gt;.csv`

Cabecera y orden EXACTOS:

```
timestamp,method,source,frame_idx,latency_ms,fps_inst,cpu_pct,ram_mb,detections
```

- timestamp: ISO 8601 consistente (local/UTC).
- method: BBoxes-HOG (literal).
- source: `webcam` o `video:&lt;ruta_relativa_o_nombre&gt;`.
- frame_idx: contador desde 0.
- latency_ms: tiempo por frame (captura→detección→escritura) medido con perf_counter.
- fps_inst: 1000.0 / latency_ms.
- cpu_pct: psutil.cpu_percent(interval=None).
- ram_mb: RSS del proceso en MB.
- detections: número de personas detectadas tras filtrar por `weight >= threshold`.

## Notas y recomendaciones de rendimiento para Raspberry Pi

- Usar `--show=False` durante las mediciones para reducir carga de GPU/CPU y evitar la capa de GUI.
- Mantener `win_stride` y `padding` moderados (p.ej., 8) y `scale` cercano a 1.05–1.1 para equilibrar precisión/velocidad.
- Si la webcam no abre, el cliente intentará índices 0..3 automáticamente y mostrará ayuda.
- El cliente imprime cada ~30 frames un resumen con FPS medio, cpu_pct y detections del último frame.

## Licencia

Uso académico/TFM. Ajustar según necesidades del proyecto.