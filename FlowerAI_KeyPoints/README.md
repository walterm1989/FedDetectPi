FlowerAI_KeyPoints

Detección de personas por puntos clave (Keypoints) usando Keypoint R-CNN (ResNet50-FPN, torchvision) en CPU, con plano de control vía Flower y registro de métricas en CSV para comparativas.

Estructura
- server.py: servidor Flower (FedAvg “stub”) que publica configuración (umbral, tamaño de entrada, draw, etc.).
- client_raspberry.py: cliente que ejecuta inferencia (CPU) sobre webcam, aplicando la configuración recibida. Si no hay servidor, funciona en modo local con valores por defecto.
- utils/metrics.py: helpers de CSV (cabecera fija).
- utils/draw.py: utilidades para dibujar esqueletos COCO sobre la imagen (opcional).
- utils/camera.py: utilidades para abrir la webcam de forma robusta (mismo interfaz que en BoundingBoxes).
- requirements.txt: dependencias mínimas (CPU).
- Metrics/: carpeta de salida para CSVs.

Prerrequisitos
- Raspberry Pi OS 64-bit (aarch64).
- Python 3.9+ recomendado.
- Cámara USB o CSI accesible como /dev/video*.
- Conectividad opcional para descargar pesos de torchvision (o bien disponer de un wheel/archivo local).

Instalación (recomendada en virtualenv)

1) Crear y activar entorno
python3 -m venv .venv
source .venv/bin/activate

2) Instalar dependencias base
pip install --upgrade pip
pip install -r FlowerAI_KeyPoints/requirements.txt

3) Instalar PyTorch/torchvision (CPU, aarch64)
En Raspberry Pi (aarch64), las ruedas oficiales pueden no estar disponibles desde PyPI. Opciones:
- Instalar desde wheels precompilados (recomendado cuando no hay red a internet):
  Descargue wheels compatibles para su versión de Python (por ejemplo, 3.9/3.10) y arquitectura aarch64:
  torch‑<ver>‑cp3x‑cp3x‑linux_aarch64.whl
  torchvision‑<ver>‑cp3x‑cp3x‑linux_aarch64.whl
  y luego:
  pip install ./torch-*.whl ./torchvision-*.whl
- Alternativa con index extra (si hay red):
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

Nota: Si no puede descargar los pesos del modelo en tiempo de ejecución, puede indicar una ruta local a un checkpoint con --weights o estableciendo TORCH_HOME para usar un caché local.

Ejecución

Servidor (plano de control):
python FlowerAI_KeyPoints/server.py --address 0.0.0.0:8080 --rounds 1 --conf 0.5 --input-size 640 --draw 0

Cliente Raspberry con Flower:
python3 FlowerAI_KeyPoints/client_raspberry.py --server <IP_DEL_PORTATIL>:8080 --cam-index 0 --duration 60 --metrics-dir FlowerAI_KeyPoints/Metrics

Cliente Raspberry sin Flower (local 60 s):
python3 FlowerAI_KeyPoints/client_raspberry.py --cam-index 0 --duration 60 --metrics-dir FlowerAI_KeyPoints/Metrics

Configuración publicada por servidor
- conf_thr (float, default 0.5): umbral de confianza.
- input_size (int, default 640): lado corto de la imagen de entrada, se reescala manteniendo aspecto.
- max_frames (int, opcional): límite de frames a procesar (además de --duration del cliente).
- draw (bool, 0/1): si se desea visualizar esqueleto sobre la imagen en el cliente.

CSV de métricas
- Directorio por defecto: FlowerAI_KeyPoints/Metrics/
- Nombre: YYYYmmdd_HHMMSS_KeyPoints-resnet50_webcam.csv
- Cabecera y orden exacto de columnas:
  timestamp,method,source,frame_idx,latency_ms,fps_inst,cpu_pct,ram_mb,detections
- Definiciones:
  - timestamp: ISO 8601, con milisegundos (datetime.now().isoformat(timespec="milliseconds")).
  - method: "KeyPoints-resnet50".
  - source: "webcam".
  - frame_idx: desde 0.
  - latency_ms: captura→modelo→postproceso.
  - fps_inst: 1000.0 / max(latency_ms, ε).
  - cpu_pct: suma por núcleo (sum(psutil.cpu_percent(percpu=True))) (puede superar 100%).
  - ram_mb: RSS del proceso en MB.
  - detections: número de personas detectadas por frame.

Ejemplo de primera línea (cabecera):
timestamp,method,source,frame_idx,latency_ms,fps_inst,cpu_pct,ram_mb,detections

Notas de rendimiento y consejos
- En Raspberry Pi 4 (CPU), Keypoint R-CNN es pesado; reduzca --input-size (por ejemplo 480) desde el servidor para bajar latencia.
- Desactive --draw para ahorrar tiempo de CPU.
- Asegúrese de usar opencv-python-headless para evitar dependencias de GUI si no necesita ventana.

Solución de problemas
- No se descargan pesos: use --weights /ruta/al/checkpoint.pth o exporte TORCH_HOME a un directorio con el modelo cacheado.
- La webcam no abre:
  - Verifique /dev/video* y permisos.
  - Pruebe con --cam-index 1,2,3.
- El cliente no conecta con Flower:
  - Se ejecuta igualmente en modo local.
  - Revise firewall/puertos y dirección --address en el servidor.

Licencia
Este proyecto se ofrece con fines demostrativos y de comparativa de rendimiento en dispositivos edge.