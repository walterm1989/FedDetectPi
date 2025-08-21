import argparse
import time
import cv2
import torch
import numpy as np
from FlowerAI.utils.model_def import build_model, load_ckpt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--no-window", action="store_true", help="Disable GUI window")
    args = parser.parse_args()

    device = "cpu"
    model = build_model(num_classes=2, freeze_backbone=False)
    ok = load_ckpt(model, args.ckpt, map_location=device)
    if not ok:
        print(f"Failed to load checkpoint from {args.ckpt}.", flush=True)
        exit(1)
    model.to(device)
    model.eval()

    cap = cv2.VideoCapture(0)
    frame_idx = 0
    prev_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t0 = time.time()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            logits = model(img)
            prob = torch.softmax(logits, dim=1)
            persons = int(prob[0, 1] > 0.5)
        t1 = time.time()
        latency_ms = int((t1 - t0) * 1000)
        fps = 1.0 / (t1 - prev_time)
        prev_time = t1
        print(f"Inference time: {latency_ms} ms | FPS: {fps:.2f} | Persons: {persons}", flush=True)
        frame_idx += 1
        if not args.no_window:
            disp = cv2.resize(frame, (448, 448))
            cv2.putText(disp, f"Persons: {persons}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow("FlowerAI Inference", disp)
            if cv2.waitKey(1) == 27:
                break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()