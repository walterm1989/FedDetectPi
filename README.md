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
