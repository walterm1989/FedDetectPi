# FlowerAI_BoundingBoxes (YOLOv4-tiny + OpenCV DNN, CPU)

Este módulo implementa detección de personas mediante Bounding Boxes en una Raspberry Pi 4 (ARM64) usando YOLOv4-tiny con OpenCV DNN en CPU. Flower AI se usa únicamente como plano de control para publicar parámetros (conf_thr, nms_thr, input_size) hacia los clientes. No se entrena ningún modelo.

Importante: Esta versión sustituye a la anterior basada en HOG. El detector aquí es YOLOv4-tiny (OpenCV DNN, CPU), no HOG.

Contenido:
- Servidor Flower (plano de control): publica parámetros dinámicos.
- Cliente Raspberry: ejecuta la detección local y reporta métricas por frame a CSV.
- Script para descargar automáticamente los assets (cfg, weights, names).
- Utilidades para cámara y métricas.

Estructura:
- FlowerAI_BoundingBoxes/
  - server.py
  - client_raspberry.py
  - download_assets.py
  - requirements.txt
  - utils/
    - camera.py
    - metrics.py
  - assets/            (se crea automáticamente si no existe)
  - Metrics/           (CSV generados)

Requisitos (Raspberry Pi 4 / ARM64)
- Python 3.9+ recomendado
- OpenCV DNN CPU (opencv-python-headless)
- Paquetes: numpy, psutil, flwr, python-dotenv (opcional)

Nota sobre visualización:
- Para mediciones de rendimiento, se recomienda ejecutar con --show desactivado (por defecto).
- Si necesitas mostrar la ventana con las detecciones, instala opencv-python (no headless) en vez de opencv-python-headless.

Instalación (Raspberry)
1) Crear entorno e instalar dependencias:
   python3 -m venv .venv &amp;&amp; source .venv/bin/activate
   pip install -r FlowerAI_BoundingBoxes/requirements.txt

2) Descargar assets (cfg, weights, names):
   python FlowerAI_BoundingBoxes/download_assets.py

3) Verifica que los archivos aparezcan en FlowerAI_BoundingBoxes/assets/:
   - yolov4-tiny.cfg
   - yolov4-tiny.weights
   - coco.names

Servidor Flower (portátil)
Inicia el servidor Flower que publicará la configuración a los clientes. No hay entrenamiento; se usa FedAvg básico como canal de control.

Ejemplo:
  python FlowerAI_BoundingBoxes/server.py \
    --address 0.0.0.0:8080 --rounds 9999 \
    --conf-thr 0.40 --nms-thr 0.45 --input-size 416

Parámetros publicados por el servidor:
- conf_thr (float): umbral de confianza, p. ej. 0.40
- nms_thr (float): umbral de NMS, p. ej. 0.45
- input_size (int): tamaño cuadrado de entrada (320, 416)
- person_class_name: "person" (filtro de detecciones)

Cliente Raspberry (detección y métricas)
El cliente conecta al servidor Flower (si está disponible) y aplica la configuración recibida dinámicamente sin reiniciar. Si el servidor no está accesible, el cliente sigue en modo local con valores por defecto, ejecuta la detección y genera los CSV de métricas igualmente.

Fuente de vídeo (CLI):
- --source [webcam|video]
- --cam-index 0 (si webcam)
- --video-path ./samples/people.mp4 (si video)
- --show (opcional; desactivado por defecto)
- --duration 60 (segundos; por defecto 60 para comparativa)
- --out-dir ./FlowerAI_BoundingBoxes/Metrics (directorio de salida CSV)
- --cfg/--weights/--names (opcional: rutas para sobrescribir assets)
- Variables de entorno: YOLO_DIR (directorio base para los assets)

Ejemplos de ejecución
Servidor (portátil):
  python FlowerAI_BoundingBoxes/server.py \
    --address 0.0.0.0:8080 --rounds 9999 \
    --conf-thr 0.40 --nms-thr 0.45 --input-size 416

Cliente (webcam, 60s, métricas):
  python FlowerAI_BoundingBoxes/client_raspberry.py \
    --server &lt;IP_PORTATIL&gt;:8080 \
    --source webcam --cam-index 0 \
    --duration 60 \
    --out-dir ./FlowerAI_BoundingBoxes/Metrics

Cliente (vídeo):
  python FlowerAI_BoundingBoxes/client_raspberry.py \
    --server &lt;IP_PORTATIL&gt;:8080 \
    --source video --video-path ./samples/people.mp4 \
    --duration 60 \
    --out-dir ./FlowerAI_BoundingBoxes/Metrics

Salida de métricas (CSV)
- Directorio: FlowerAI_BoundingBoxes/Metrics/
- Nombre de archivo:
  - YYYYMMDD_HHMMSS_BBoxes-YOLOv4tiny_webcam.csv
  - YYYYMMDD_HHMMSS_BBoxes-YOLOv4tiny_video-&lt;nombre&gt;.csv
- Cabecera y orden exactos:
  timestamp,method,source,frame_idx,latency_ms,fps_inst,cpu_pct,ram_mb,detections

Campo "method" debe ser exactamente: BBoxes-YOLOv4tiny

Recomendaciones de rendimiento
- Ejecuta sin --show para medir FPS/latencia de forma realista.
- Asegúrate de usar opencv-python-headless en Raspberry si no necesitas ventana.
- Cierra otras aplicaciones para evitar interferencias en CPU.

Scripts de ayuda
- run_server.sh: ejemplo para lanzar el servidor.
- run_client.sh: ejemplo para lanzar el cliente (webcam).

Licencias y créditos
- YOLOv4-tiny: pesos y cfg de la comunidad (AlexeyAB/darknet).
- COCO names: lista de clases del conjunto COCO.
- OpenCV DNN: backend de inferencia en CPU.