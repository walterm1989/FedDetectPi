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
from datetime import datetime

# Rutas de los archivos de modelo y clases
RUTA_PESOS = 'yolov4-tiny.weights'
RUTA_CFG = 'yolov4-tiny.cfg'
RUTA_NOMBRES = 'coco.names'
ARCHIVO_CSV = 'metricas_boundingboxes.csv'

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

def medir_recursos():
    """
    Retorna el uso actual de CPU (%) y RAM (MB) usando psutil.
    """
    cpu = psutil.cpu_percent(interval=None)
    ram = psutil.virtual_memory().used / (1024*1024)
    return round(cpu, 1), round(ram, 1)

def guardar_csv(ruta_csv, datos, existe_archivo):
    """
    Escribe una línea de métricas en el archivo CSV.
    Si el archivo no existe, escribe la cabecera.
    """
    escribir_cabecera = not existe_archivo
    with open(ruta_csv, 'a', newline='') as f:
        escritor = csv.writer(f)
        if escribir_cabecera:
            escritor.writerow(['timestamp', 'latencia_ms', 'cpu_percent', 'ram_mb'])
        escritor.writerow(datos)

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
    # Cargar modelo y capas de salida
    red, capas_salida, nombres_clases = cargar_modelo()

    # Iniciar captura de cámara
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: No se pudo abrir la cámara. Verifica que esté conectada y disponible.")
        exit(1)
    
    print("Cámara iniciada. Comenzando detección de personas (120 segundos o pulsa 'q' para salir)...")
    
    # Preparación de archivo CSV
    ruta_csv = os.path.join(os.path.dirname(__file__), ARCHIVO_CSV)
    existe_archivo = os.path.isfile(ruta_csv)
    
    tiempo_inicio = time.time()
    while True:
        tiempo_ahora = time.time()
        if tiempo_ahora - tiempo_inicio > 120:
            print("Tiempo máximo alcanzado (120 segundos). Finalizando.")
            break

        ret, frame = cap.read()
        if not ret:
            print("ERROR: No se pudo leer un frame de la cámara.")
            break

        t0 = time.time()
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
        latencia_ms = int((time.time() - t0)*1000)
        cpu_percent, ram_mb = medir_recursos()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        guardar_csv(ruta_csv, [timestamp, latencia_ms, cpu_percent, ram_mb], existe_archivo)
        existe_archivo = True  # Solo se escribe cabecera la primera vez

        # Mostrar frame
        texto_metricas = f"Latencia: {latencia_ms} ms | CPU: {cpu_percent}% | RAM: {ram_mb} MB"
        cv2.putText(frame, texto_metricas, (10, alto-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
        cv2.imshow('Detección de Personas - YOLOv4-tiny', frame)
        
        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Finalización anticipada por el usuario.")
            break

    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()
    print("Recursos liberados. ¡Hasta luego!")

if __name__ == '__main__':
    main()