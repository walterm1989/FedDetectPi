"""
Webcam inference client for FlowerAI.
"""

import os
import sys
import cv2
import time
import glob
import torch
import numpy as np
from FlowerAI.utils.model_def import build_model, load_ckpt
import argparse

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints")

def find_latest_checkpoint():
    # List all global_round_*.pt files and latest_global.pt
    files = glob.glob(os.path.join(CHECKPOINT_DIR, "global_round_*.pt"))
    files.sort(key=os.path.getmtime, reverse=True)
    if os.path.exists(os.path.join(CHECKPOINT_DIR, "latest_global.pt")):
        return os.path.join(CHECKPOINT_DIR, "latest_global.pt")
    elif files:
        return files[0]
    return None

def preprocess(frame):
    # Simple preprocessing: resize and normalize
    img = cv2.resize(frame, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # BCHW
    return tensor

def main():
    parser = argparse.ArgumentParser(description="FlowerAI webcam inference")
    parser.add_argument("--no-window", action="store_true", help="Disable display window")
    args = parser.parse_args()

    ckpt_path = find_latest_checkpoint()
    if not ckpt_path:
        print("No checkpoint found in FlowerAI/checkpoints/")
        sys.exit(1)

    model = build_model(num_classes=2, freeze_backbone=False)
    load_ckpt(model, ckpt_path, strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        sys.exit(1)

    last_time = time.time()
    frame_count = 0
    total_latency = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            t0 = time.time()
            inp = preprocess(frame).to(device)
            with torch.no_grad():
                logits = model(inp)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                pred = int(np.argmax(probs))
            t1 = time.time()
            latency_ms = (t1 - t0) * 1000
            total_latency += latency_ms
            frame_count += 1
            fps = frame_count / (t1 - last_time + 1e-6)
            # Parseable output: timestamp, latency_ms, fps, persons, probs
            print(f"[INFER] time={time.strftime('%Y-%m-%d %H:%M:%S')} "
                  f"latency_ms={latency_ms:.1f} fps={fps:.2f} persons={pred} probs={probs.tolist()}",
                  flush=True)
            # Draw result
            if not args.no_window:
                label = "PERSON" if pred == 1 else "NO_PERSON"
                disp_txt = f"{label} ({probs[1]:.2f})" if pred == 1 else f"{label} ({probs[0]:.2f})"
                cv2.putText(frame, disp_txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0) if pred else (0,0,255), 2)
                cv2.imshow("FlowerAI Inference", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    finally:
        cap.release()
        if not args.no_window:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()