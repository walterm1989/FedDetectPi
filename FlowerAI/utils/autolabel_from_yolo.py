# === CONFIG (editar aquí) ===
CAM_INDEX        = 0          # índice de cámara
CAPTURE_SIZE     = (320, 240) # resolución de captura (w, h) para RPi
DURATION_SEC     = 300        # tiempo máx de muestreo
FRAME_SKIP       = 2          # saltar N frames entre detecciones (reduce carga)
SAVE_EVERY_N     = 1          # guardar 1 de cada N frames evaluados
CONF_THRESH      = 0.40       # umbral de confianza para 'person'
NMS_THRESH       = 0.30       # NMS
OUTPUT_SIZE      = (224, 224) # tamaño de guardado (w, h)
SHOW_WINDOW      = True       # False para headless
SEARCH_YOLO_PATHS = [
    "BoudingBoxes", "BoundingBoxes", ".", "FlowerAI/utils"
] # rutas relativas donde buscar cfg/weights/names
CFG_NAME   = "yolov4-tiny.cfg"
WEIGHTS    = "yolov4-tiny.weights"
NAMES_FILE = "coco.names"
OUT_DIR_PERSON    = "FlowerAI/data/person"
OUT_DIR_NO_PERSON = "FlowerAI/data/no_person"
# === FIN CONFIG ===

import os
import sys
import cv2
import time
from datetime import datetime

# -------------- Utility: Find YOLO files --------------
def find_yolo_files():
    found = {"cfg": None, "weights": None, "names": None}
    for base in SEARCH_YOLO_PATHS:
        c = os.path.join(base, CFG_NAME)
        w = os.path.join(base, WEIGHTS)
        n = os.path.join(base, NAMES_FILE)
        if found["cfg"] is None and os.path.isfile(c):
            found["cfg"] = c
        if found["weights"] is None and os.path.isfile(w):
            found["weights"] = w
        if found["names"] is None and os.path.isfile(n):
            found["names"] = n
        if all(found.values()):
            break
    return found

def print_wget_commands(missing):
    print("\nMissing YOLO files! Download them with the following commands:\n")
    baseurls = {
        "cfg": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg",
        "weights": "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights",
        "names": "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
    }
    for key, url in baseurls.items():
        if missing[key] is None:
            fname = {"cfg": CFG_NAME, "weights": WEIGHTS, "names": NAMES_FILE}[key]
            print(f"wget -O {fname} {url}")
    print("\nPlace the downloaded files in one of the following folders:")
    for p in SEARCH_YOLO_PATHS:
        print(f"  {os.path.abspath(p)}")
    print()

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def unique_filename(label, seq, dt):
    # dt: datetime object
    tstamp = dt.strftime("%Y%m%d_%H%M%S_%f")[:-3] # %f gives microseconds, keep only milliseconds
    return f"{label}/{tstamp}_{seq:04d}.jpg"

def main():
    yolo_files = find_yolo_files()
    missing = {k: v for k, v in yolo_files.items() if v is None}
    if missing:
        print_wget_commands(yolo_files)
        print("ERROR: One or more YOLO model files are missing. Exiting.")
        sys.exit(1)

    # Load class names
    with open(yolo_files["names"], "r") as f:
        class_names = [line.strip() for line in f.readlines()]

    if len(class_names) == 0 or class_names[0].lower() != "person":
        print("ERROR: COCO class 0 is not 'person'. Check your names file.")
        sys.exit(1)

    # Load YOLO model
    try:
        net = cv2.dnn.readNetFromDarknet(yolo_files["cfg"], yolo_files["weights"])
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    except Exception as e:
        print(f"ERROR: Failed to load YOLOv4-tiny: {e}")
        sys.exit(1)

    out_layers = net.getUnconnectedOutLayersNames()

    ensure_dir(OUT_DIR_PERSON)
    ensure_dir(OUT_DIR_NO_PERSON)

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"ERROR: Could not open webcam at index {CAM_INDEX}")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_SIZE[1])

    # Warm up camera (skip a few frames)
    print(f"Warming up camera...")
    for _ in range(10):
        ret, _ = cap.read()
        if not ret:
            print("ERROR: Failed to read from webcam during warmup.")
            cap.release()
            sys.exit(1)
        time.sleep(0.02)

    print("Starting auto-labeling. Press 'q' to quit early.")
    start_time = time.perf_counter()
    processed = 0
    saved = 0
    saved_person = 0
    saved_no_person = 0
    frame_idx = 0
    seq = 0

    try:
        while True:
            now = time.perf_counter()
            elapsed = now - start_time
            if elapsed > DURATION_SEC:
                print(f"\nTime limit of {DURATION_SEC}s reached.")
                break

            ret, frame = cap.read()
            if not ret:
                print("WARNING: Failed to read frame. Skipping.")
                continue
            frame_idx += 1
            if frame_idx % FRAME_SKIP != 0:
                continue

            # Prepare input blob
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416,416), swapRB=True, crop=False)
            net.setInput(blob)
            outs = net.forward(out_layers)

            frame_h, frame_w = frame.shape[:2]
            boxes = []
            confidences = []

            for out in outs:
                for det in out:
                    scores = det[5:]
                    class_id = int(scores.argmax())
                    conf = scores[class_id]
                    if class_id == 0 and conf > CONF_THRESH:
                        center_x = int(det[0] * frame_w)
                        center_y = int(det[1] * frame_h)
                        w = int(det[2] * frame_w)
                        h = int(det[3] * frame_h)
                        x = int(center_x - w/2)
                        y = int(center_y - h/2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(conf))
            indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESH, NMS_THRESH)
            person_detections = []
            if len(indices) > 0:
                for i in indices.flatten():
                    person_detections.append(boxes[i])

            label = "person" if len(person_detections) > 0 else "no_person"
            processed += 1

            # Save frame if required
            if processed % SAVE_EVERY_N == 0:
                seq += 1
                out_dir = OUT_DIR_PERSON if label == "person" else OUT_DIR_NO_PERSON
                ensure_dir(out_dir)
                dt = datetime.now()
                # Save as per required pattern: person/YYYYmmdd_HHMMSS_mmm_<seq>.jpg
                fname = unique_filename(label, seq, dt)
                out_path = os.path.join(out_dir, os.path.basename(fname))
                to_save = cv2.resize(frame, OUTPUT_SIZE)
                cv2.imwrite(out_path, to_save, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                saved += 1
                if label == "person":
                    saved_person += 1
                else:
                    saved_no_person += 1

            # Show window
            if SHOW_WINDOW:
                disp = frame.copy()
                for (x, y, w, h) in person_detections:
                    cv2.rectangle(disp, (x, y), (x+w, y+h), (0,255,0), 2)
                    cv2.putText(disp, "person", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.putText(disp, f"Labeled: {label}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255) if label=="no_person" else (0,255,0), 2)
                cv2.putText(disp, f"Processed: {processed}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                cv2.putText(disp, f"Saved: {saved} (person: {saved_person}, no_person: {saved_no_person})", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,0), 2)
                cv2.putText(disp, f"Time: {int(elapsed)}s / {DURATION_SEC}s", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
                cv2.imshow("YOLOv4-tiny AutoLabel", disp)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quitting due to user keypress.")
                    break

    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting...")
    finally:
        cap.release()
        if SHOW_WINDOW:
            cv2.destroyAllWindows()
        print("\n======= Auto-Labeling Run Complete =======")
        print(f"Total frames processed: {processed}")
        print(f"Total frames saved: {saved}")
        print(f"  person:    {saved_person}")
        print(f"  no_person: {saved_no_person}")
        if saved > 0:
            ratio = saved_person / saved
            print(f"Person/no_person ratio: {ratio:.2f}")
        else:
            print("No frames saved.")
        print("=========================================\n")

if __name__ == "__main__":
    main()