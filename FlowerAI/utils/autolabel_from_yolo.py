# ========================== CONFIG ==========================
CONFIG = {
    # List of directories (relative to this script) to search for YOLO files
    "SEARCH_YOLO_PATHS": [
        "../yolo",
        "../../yolo",
        "../models/yolo",
        "../../models/yolo",
        "./yolo",
        ".",
    ],

    # YOLO files (relative to search paths)
    "YOLO_CFG_FILE": "yolov4-tiny.cfg",
    "YOLO_WEIGHTS_FILE": "yolov4-tiny.weights",
    "YOLO_NAMES_FILE": "coco.names",

    # Webcam settings
    "CAM_INDEX": 0,
    "CAM_WIDTH": 640,
    "CAM_HEIGHT": 480,

    # Processing
    "CONF_THRESH": 0.3,
    "NMS_THRESH": 0.4,
    "FRAME_SKIP": 3,          # Only process every Nth frame
    "WARMUP_FRAMES": 15,      # Number of frames to "warm up" camera

    # Output
    "OUT_DIR_PERSON": "autolabel_out/person",
    "OUT_DIR_NO_PERSON": "autolabel_out/no_person",
    "OUTPUT_SIZE": (224, 224),
    "SAVE_EVERY_N": 2,        # Save every Nth processed frame

    # Runtime
    "SHOW_WINDOW": True,
    "TIME": 60,               # Seconds to run main loop
}
# ======================== END CONFIG ========================

import os
import sys
import cv2
import time
import threading
from datetime import datetime

# -------------- Utility: Find YOLO files --------------
def find_yolo_files():
    paths = CONFIG["SEARCH_YOLO_PATHS"]
    cfg = CONFIG["YOLO_CFG_FILE"]
    weights = CONFIG["YOLO_WEIGHTS_FILE"]
    names = CONFIG["YOLO_NAMES_FILE"]

    found = {"cfg": None, "weights": None, "names": None}
    for base in paths:
        c = os.path.join(base, cfg)
        w = os.path.join(base, weights)
        n = os.path.join(base, names)
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
            fname = CONFIG[f"YOLO_{key.upper()}_FILE"]
            print(f"wget -O {fname} {url}")
    print("\nPlace the downloaded files in one of the following folders:")
    for p in CONFIG["SEARCH_YOLO_PATHS"]:
        print(f"  {os.path.abspath(p)}")
    print()
# -------------------------------------------------------

# -------------- Utility: Create Output Dirs ------------
def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

# -------------- Utility: Timestamp ---------------------
def unique_filename(label, seq):
    now = datetime.now()
    tstamp = now.strftime("%Y%m%d_%H%M%S_%f")
    return f"{label}_{tstamp}_seq{seq}.jpg"

# -------------- Main Script ----------------------------
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

    # Check that "person" is present as class id 0
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

    # Get output layer names
    out_layers = net.getUnconnectedOutLayersNames()

    # Prepare output dirs
    out_dir_person = CONFIG["OUT_DIR_PERSON"]
    out_dir_no_person = CONFIG["OUT_DIR_NO_PERSON"]
    ensure_dir(out_dir_person)
    ensure_dir(out_dir_no_person)

    # Open webcam
    cap = cv2.VideoCapture(CONFIG["CAM_INDEX"])
    if not cap.isOpened():
        print(f"ERROR: Could not open webcam at index {CONFIG['CAM_INDEX']}")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["CAM_WIDTH"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["CAM_HEIGHT"])

    # Warm up camera
    print(f"Warming up camera for {CONFIG['WARMUP_FRAMES']} frames...")
    for _ in range(CONFIG['WARMUP_FRAMES']):
        ret, _ = cap.read()
        if not ret:
            print("ERROR: Failed to read from webcam during warmup.")
            cap.release()
            sys.exit(1)
        time.sleep(0.02)  # Small delay

    print("Starting auto-labeling. Press 'q' to quit early.")
    start_time = time.time()
    elapsed = 0
    processed = 0
    saved = 0
    saved_person = 0
    saved_no_person = 0
    frame_idx = 0
    seq = 0

    show_win = CONFIG["SHOW_WINDOW"]
    save_every = CONFIG["SAVE_EVERY_N"]
    output_size = CONFIG["OUTPUT_SIZE"]
    conf_thresh = CONFIG["CONF_THRESH"]
    nms_thresh = CONFIG["NMS_THRESH"]
    frame_skip = CONFIG["FRAME_SKIP"]
    max_time = CONFIG["TIME"]

    try:
        while True:
            now = time.time()
            elapsed = now - start_time
            if elapsed > max_time:
                print(f"\nTime limit of {max_time}s reached.")
                break

            ret, frame = cap.read()
            if not ret:
                print("WARNING: Failed to read frame. Skipping.")
                continue
            frame_idx += 1
            if frame_idx % frame_skip != 0:
                continue

            # Prepare input blob
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416,416), swapRB=True, crop=False)
            net.setInput(blob)
            outs = net.forward(out_layers)

            # Gather detections
            frame_h, frame_w = frame.shape[:2]
            boxes = []
            confidences = []
            class_ids = []

            for out in outs:
                for det in out:
                    scores = det[5:]
                    class_id = int(scores.argmax())
                    conf = scores[class_id]
                    if class_id == 0 and conf > conf_thresh:
                        # Person detected
                        center_x = int(det[0] * frame_w)
                        center_y = int(det[1] * frame_h)
                        w = int(det[2] * frame_w)
                        h = int(det[3] * frame_h)
                        x = int(center_x - w/2)
                        y = int(center_y - h/2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(conf))
                        class_ids.append(class_id)
            # Apply NMS
            indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, nms_thresh)
            person_detections = []
            if len(indices) > 0:
                for i in indices.flatten():
                    person_detections.append(boxes[i])

            label = "person" if len(person_detections) > 0 else "no_person"
            processed += 1

            # Save frame if required
            if processed % save_every == 0:
                seq += 1
                out_dir = out_dir_person if label == "person" else out_dir_no_person
                ensure_dir(out_dir)
                fname = unique_filename(label, seq)
                out_path = os.path.join(out_dir, fname)
                to_save = cv2.resize(frame, output_size)
                cv2.imwrite(out_path, to_save, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                saved += 1
                if label == "person":
                    saved_person += 1
                else:
                    saved_no_person += 1

            # Show window
            if show_win:
                disp = frame.copy()
                # Draw detections
                for (x, y, w, h) in person_detections:
                    cv2.rectangle(disp, (x, y), (x+w, y+h), (0,255,0), 2)
                    cv2.putText(disp, "person", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.putText(disp, f"Labeled: {label}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255) if label=="no_person" else (0,255,0), 2)
                cv2.putText(disp, f"Processed: {processed}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                cv2.putText(disp, f"Saved: {saved} (person: {saved_person}, no_person: {saved_no_person})", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,0), 2)
                cv2.putText(disp, f"Time: {int(elapsed)}s / {max_time}s", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
                cv2.imshow("YOLOv4-tiny AutoLabel", disp)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quitting due to user keypress.")
                    break

    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting...")
    finally:
        cap.release()
        if show_win:
            cv2.destroyAllWindows()
        # Print summary
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