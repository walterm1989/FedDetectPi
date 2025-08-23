# FedDetectPi
Sistema de detección de personas basado en aprendizaje federado con visión artificial distribuida en Raspberry Pi y portátil usando Flower AI.

---

## Raspberry Pi defaults

`inferencia_raspberry.py` is pre-configured with optimized defaults for running real-time keypoint detection on a Raspberry Pi 4. The main constants are defined at the top of the script:

- `DEFAULT_CAP_WIDTH` / `DEFAULT_CAP_HEIGHT`: Camera capture resolution (default: 320x240).
- `DEFAULT_PROC_SCALE`: Factor to downscale frames before inference (default: 0.5). Lower for faster inference, higher for more detail.
- `DEFAULT_SKIP_FRAMES`: Number of frames to skip between inferences (default: 2, i.e., run inference every 3rd frame).
- `DEFAULT_HALF_PRECISION`: Whether to use float16 ("half precision") on supported hardware (default: True).
- `TARGET_LATENCY_MS`: Target latency per inference call, in milliseconds (default: 200 ms). The script adaptively adjusts `skip_frames` to try to meet this.

You can override these defaults via command-line arguments. For example:
```
python3 inferencia_raspberry.py --width 640 --height 480 --proc-scale 0.8 --skip-frames 0 --no-half --target-latency-ms 300
```

To force or disable half-precision, use `--half` or `--no-half`.

Tweak these parameters to balance detection quality and real-time speed, depending on your Raspberry Pi's performance and workload.

---

## Flower AI (federado)

Este proyecto utiliza Flower AI para entrenamiento federado distribuido entre Raspberry Pi y servidor portátil.

### Cómo iniciar el servidor

```bash
python -m FlowerAI.server
```

### Cómo iniciar el cliente en Raspberry Pi

```bash
export FLOWER_SERVER_ADDR=&lt;IP:PORT&gt;
python -m FlowerAI.client.rpi
```
Reemplaza <IP:PORT> por la dirección y puerto del servidor (por ejemplo: 192.168.1.10:8080).

### Cómo ejecutar el medidor federado

```bash
python -m FlowerAI.client.medidor_federado
```

### Directorio de datos y checkpoints

- Los datos de entrenamiento y validación deben ubicarse en: `FlowerAI/data/{person,no_person}`
- Los checkpoints de los modelos federados se guardan en: `FlowerAI/checkpoints/`

### Métricas y formato de datos

- El script `Metrics/plot_metrics.py` admite métricas parseables en formato de línea y estándar CSV. 
- Asegúrate de que los resultados generados sean compatibles con el formato esperado por este script para graficar y analizar las métricas.

# FlowerAI Server

## Setup (portátil)
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
pip install -r requirements.txt
python -m FlowerAI.server
```

Variables opcionales:
- FLOWER_SERVER_ADDRESS (por defecto 0.0.0.0:8080)
- FLOWER_NUM_ROUNDS (por defecto 3)

## Cliente Raspberry

Para instalar las dependencias necesarias para el cliente en Raspberry Pi:

```bash
pip install -r requirements.txt
```

### Pasos en la Raspberry Pi (64-bit)
```bash
git clone <tu-repo>
cd <tu-repo>
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Establecer la IP del portátil y puerto del servidor (ej.: 192.168.1.50:8080)
export FLOWER_SERVER_ADDRESS=192.168.1.50:8080

# Ejecutar cliente
python -m FlowerAI.client.raspberry_client
```

### Métricas automáticas (CPU/RAM) en RPi

El cliente registra CPU (%) y RAM (MB) cada 1 s durante 60 s en `Metrics/flower_rpi_metrics_YYYYMMDD_HHMMSS.csv`.

Variables opcionales:
- `METRICS_SECONDS` (por defecto `60`)
- `METRICS_INTERVAL` (por defecto `1.0`)
- `METRICS_DIR` (por defecto `Metrics`)
- `METRICS_CSV` (ruta completa si quieres nombrar el archivo manualmente)

Cómo obtener la IP del portátil:
- Windows: ipconfig → “IPv4 Address”.
- Linux/Mac: ip addr o ifconfig.

Con el servidor activo, deberías ver logs tipo:
- En el servidor: “Requesting initial parameters from one random client”
- En la Raspberry: “[FlowerAI][RPi] Conectando…”, “Fit…”, “Evaluate…”

### Métricas automáticas (CPU/RAM) en RPi

El cliente registra CPU (%) y RAM (MB) cada 1 s durante 60 s en `Metrics/flower_rpi_metrics_YYYYMMDD_HHMMSS.csv`.

Variables opcionales:
- `METRICS_SECONDS` (por defecto `60`)
- `METRICS_INTERVAL` (por defecto `1.0`)
- `METRICS_DIR` (por defecto `Metrics`)
- `METRICS_CSV` (ruta completa si quieres nombrar el archivo manualmente)

Cómo obtener la IP del portátil:
- Windows: ipconfig → “IPv4 Address”.
- Linux/Mac: ip addr o ifconfig.

Con el servidor activo, deberías ver logs tipo:
- En el servidor: “Requesting initial parameters from one random client”
- En la Raspberry: “[FlowerAI][RPi] Conectando…”, “Fit…”, “Evaluate…”

Cómo obtener la IP del portátil:
- Windows: ipconfig → “IPv4 Address”.
- Linux/Mac: ip addr o ifconfig.

Con el servidor activo, deberías ver logs tipo:
- En el servidor: “Requesting initial parameters from one random client”
- En la Raspberry: “[FlowerAI][RPi] Conectando…”, “Fit…”, “Evaluate…”

### Métricas automáticas (CPU/RAM) en RPi

El cliente registra CPU (%) y RAM (MB) cada 1 s durante 60 s en `Metrics/flower_rpi_metrics_YYYYMMDD_HHMMSS.csv`.

Variables opcionales:
- `METRICS_SECONDS` (por defecto `60`)
- `METRICS_INTERVAL` (por defecto `1.0`)
- `METRICS_DIR` (por defecto `Metrics`)
- `METRICS_CSV` (ruta completa si quieres nombrar el archivo manualmente)

```

## Bounding Boxes (RPi) con OpenCV HOG

### Ejecución (RPi)
```bash
cd BoudingBoxes
python webcam_bb.py

Variables útiles:
- METRICS_SECONDS (por defecto 60)
- METRICS_INTERVAL (por defecto 1.0)
- METRICS_CSV (ruta del CSV; por defecto Metrics/bb_rpi_metrics_*.csv)
- CAM_INDEX (por defecto 0), CAM_W/CAM_H/CAM_FPS
- SHOW=1 para visualizar ventana (si hay entorno gráfico)
```

Rama principal y commit
```bash
git checkout -B main && git pull --rebase
git add .
git commit -m "feat(bb-rpi): webcam HOG person detector + FPS/CPU/RAM CSV"
git push -u origin main
```