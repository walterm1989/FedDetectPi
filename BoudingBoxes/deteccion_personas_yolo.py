# --------------------------------------------------------------------------------
# Script: deteccion_personas_yolo.py
# Descripción:
#   Detección de personas en tiempo real usando YOLOv4-tiny y OpenCV DNN en Raspberry Pi 4.
#   Muestra el vídeo de la cámara local con bounding boxes sobre personas detectadas.
#   Registra métricas de latencia, uso de CPU y RAM en un archivo CSV.
#
# Dependencias:
#   pip install opencv-python numpy psutil
#
# Descarga de pesos y configuración (ejecutar en el mismo directorio que este script):
#   wget https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights
#   wget https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4-tiny.cfg?raw=true -O yolov4-tiny.cfg
#   wget https://github.com/pjreddie/darknet/blob/master/data/coco.names?raw=true -O coco.names
# --------------------------------------------------------------------------------

import cv2
import numpy as np
import psutil
import time
import csv
import os
import argparse
from datetime import datetime

# Rutas de los archivos de modelo y clases
RUTA_PESOS = 'yolov4-tiny.weights'
RUTA_CFG = 'yolov4-tiny.cfg'
RUTA_NOMBRES = 'coco.names'
# ARCHIVO_CSV is deprecated in favor of standardized CSV path below

# Parámetros de detección
UMBRAL_CONFIDENCIA = 0.4
UMBRAL_NMS = 0.3

def cargar_modelo():
    """
    Carga la red YOLOv4-tiny usando OpenCV DNN y retorna la red y las capas de salida.
    Maneja errores explicativos si faltan archivos.
    """
    # Revisar que existen los archivos necesarios
    for ruta, descripcion in [(RUTA_CFG, "configuración (.cfg)"),
                              (RUTA_PESOS, "pesos (.weights)"),
                              (RUTA_NOMBRES, "nombres de clases (.names)")]:
        if not os.path.isfile(ruta):
            print(f"ERROR: Falta el archivo {descripcion}: '{ruta}'.\n"
                  f"Descárgalo siguiendo las instrucciones al inicio del script.")
            exit(1)
    # Cargar nombres de clases
    with open(RUTA_NOMBRES, 'r') as f:
        nombres_clases = [linea.strip() for linea in f.readlines()]
    # Cargar red
    try:
        red = cv2.dnn.readNet(RUTA_PESOS, RUTA_CFG)
        red.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        red.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    except Exception as e:
        print(f"ERROR al cargar la red YOLO: {e}")
        exit(1)
    # Obtener nombres de capas de salida
    capas_salida = red.getUnconnectedOutLayersNames()
    return red, capas_salida, nombres_clases

def procesar_detecciones(salidas, ancho, alto, nombres_clases):
    """
    Procesa las detecciones de YOLO, filtrando solo la clase 'person' (id=0 en COCO).
    Aplica NMS y retorna listas de cajas y confidencias.
    """
    cajas = []
    confidencias = []
    indices_clase = []
    for salida in salidas:
        for deteccion in salida:
            puntajes = deteccion[5:]
            id_clase = np.argmax(puntajes)
            confianza = puntajes[id_clase]
            # Solo personas (clase 0 en COCO)
            if id_clase == 0 and confianza > UMBRAL_CONFIDENCIA:
                caja = deteccion[0:4] * np.array([ancho, alto, ancho, alto])
                centro_x, centro_y, w, h = caja.astype('int')
                x = int(centro_x - w/2)
                y = int(centro_y - h/2)
                cajas.append([x, y, int(w), int(h)])
                confidencias.append(float(confianza))
                indices_clase.append(id_clase)
    # Aplicar NMS para eliminar solapamientos
    indices = cv2.dnn.NMSBoxes(cajas, confidencias, UMBRAL_CONFIDENCIA, UMBRAL_NMS)
    cajas_filtradas = []
    confidencias_filtradas = []
    for i in indices:
        i = i[0] if isinstance(i, (list, np.ndarray)) else i
        cajas_filtradas.append(cajas[i])
        confidencias_filtradas.append(confidencias[i])
    return cajas_filtradas, confidencias_filtradas

def medir_recursos(proc=None):
    """
    Retorna el uso actual de CPU (%) y RAM (MB) usando psutil.Process.
    """
    if proc is None:
        proc = psutil.Process()
    cpu = proc.cpu_percent(interval=None)
    ram = proc.memory_info().rss / (1024 * 1024)
    return round(cpu, 1), round(ram, 1)

def guardar_csv(ruta_csv, datos, header):
    """
    Escribe una línea de métricas en el archivo CSV.
    Si el archivo no existe, escribe la cabecera.
    Fuerza flush tras cada línea.
    """
    archivo_existe = os.path.isfile(ruta_csv)
    with open(ruta_csv, 'a', newline='') as f:
        escritor = csv.writer(f)
        if not archivo_existe:
            escritor.writerow(header)
        escritor.writerow(datos)
        f.flush()

def dibujar_cajas(imagen, cajas, confidencias):
    """
    Dibuja bounding boxes y etiquetas de confianza sobre la imagen.
    """
    for caja, conf in zip(cajas, confidencias):
        x, y, w, h = caja
        cv2.rectangle(imagen, (x, y), (x+w, y+h), (0, 255, 0), 2)
        etiqueta = f"Persona: {int(conf*100)}%"
        cv2.putText(imagen, etiqueta, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

def main():
    parser = argparse.ArgumentParser(description="Detección de personas con YOLOv4-tiny y métricas estandarizadas a CSV.")
    parser.add_argument('--duration', type=int, default=120, help='Duración máxima en segundos (default 120)')
    parser.add_argument('--source', type=str, default='webcam', help="Fuente de video (default 'webcam')")
    args = parser.parse_args()

    duration = args.duration
    source = args.source

    # Parámetros CSV/Metrics
    method = 'BBoxes-YOLOv4tiny'
    ts_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_dir = os.path.join('Metrics', 'raw')
    os.makedirs(csv_dir, exist_ok=True)
    csv_name = f"{ts_str}_{method}_{source}.csv"
    ruta_csv = os.path.join(csv_dir, csv_name)
    header = [
        'timestamp',
        'method',
        'source',
        'frame_idx',
        'latency_ms',
        'fps_inst',
        'cpu_pct',
        'ram_mb',
        'detections',
    ]
    # Definir el nombre de ventana como constante
    WINDOW_NAME = "Detección de Personas - YOLOv4-tiny"

    # Cargar modelo y capas de salida
    red, capas_salida, nombres_clases = cargar_modelo()

    # Iniciar captura de cámara
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: No se pudo acceder a la cámara. Verifica que esté conectada.")
        return

    # Crear la ventana de visualización UNA sola vez
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    print(f"Cámara iniciada. Comenzando detección de personas ({duration} segundos o pulsa 'q' para salir)...")
    tiempo_inicio = time.time()
    proc = psutil.Process()
    frame_idx = 0

    try:
        while True:
            tiempo_ahora = time.time()
            if tiempo_ahora - tiempo_inicio > duration:
                print(f"Tiempo máximo alcanzado ({duration} segundos). Finalizando.")
                break

            ret, frame = cap.read()
            if not ret:
                print("ERROR: No se pudo leer un frame de la cámara.")
                break

            t0 = time.perf_counter()
            alto, ancho = frame.shape[:2]

            # Preprocesamiento para DNN
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
            red.setInput(blob)
            salidas = red.forward(capas_salida)

            # Procesar detecciones
            cajas, confidencias = procesar_detecciones(salidas, ancho, alto, nombres_clases)

            # Dibujar resultados en la imagen
            dibujar_cajas(frame, cajas, confidencias)

            # Métricas
            t1 = time.perf_counter()
            latencia_ms = (t1 - t0) * 1000
            fps_inst = 1000.0 / latencia_ms if latencia_ms > 0 else 0.0
            cpu_pct, ram_mb = medir_recursos(proc)
            timestamp = datetime.now().isoformat(timespec='milliseconds')
            detections = len(cajas)

            # CSV registro
            fila = [
                timestamp,
                method,
                source,
                frame_idx,
                round(latencia_ms, 2),
                round(fps_inst, 2),
                cpu_pct,
                ram_mb,
                detections,
            ]
            guardar_csv(ruta_csv, fila, header)
            frame_idx += 1

            # Mostrar frame
            texto_metricas = f"Latencia: {int(latencia_ms)} ms | FPS: {fps_inst:.1f} | CPU: {cpu_pct}% | RAM: {ram_mb} MB | Det: {detections}"
            cv2.putText(frame, texto_metricas, (10, alto-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
            cv2.imshow(WINDOW_NAME, frame)

            # Salir con 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Finalización anticipada por el usuario.")
                break
    finally:
        # Liberar recursos siempre, incluso si hay excepción
        cap.release()
        cv2.destroyAllWindows()
        print("Recursos liberados. ¡Hasta luego!")

if __name__ == '__main__':
    main()