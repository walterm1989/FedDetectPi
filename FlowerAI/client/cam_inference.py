import os
import sys
import time
import torch
import numpy as np
import cv2
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Webcam inference with FlowerAI model")
    parser.add_argument('--no-window', action='store_true', help="Do not display the OpenCV window")
    return parser.parse_args()

def find_checkpoint(ckpt_dir):
    latest_ckpt = os.path.join(ckpt_dir, "latest_global.pt")
    if os.path.exists(latest_ckpt):
        return latest_ckpt
    # Fallback: find newest "global_round_*.pt"
    candidates = []
    for f in os.listdir(ckpt_dir):
        if f.startswith("global_round_") and f.endswith(".pt"):
            candidates.append(os.path.join(ckpt_dir, f))
    if not candidates:
        raise FileNotFoundError("No model checkpoint found in {}".format(ckpt_dir))
    # Sort by modification time, newest last
    candidates.sort(key=lambda x: os.path.getmtime(x))
    return candidates[-1]

def main():
    args = get_args()

    # Import model utils
    utils_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "utils")
    sys.path.append(utils_dir)
    from model_utils import build_model  # assuming utils/model_utils.py

    # Load model
    ckpt_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
    ckpt_path = find_checkpoint(ckpt_dir)
    model = build_model()
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()
    model.to("cpu")

    # Set up webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # Main loop
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        # Resize and center crop to 224x224 (assume model expects 224)
        img = cv2.resize(frame, (256, 256))
        h, w, _ = img.shape
        startx = w//2 - (224//2)
        starty = h//2 - (224//2)
        img = img[starty:starty+224, startx:startx+224]

        # Convert BGR to RGB, normalize, to float32
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - mean) / std
        img = np.transpose(img, (2, 0, 1))  # HWC to CHW
        img_tensor = torch.from_numpy(img).unsqueeze(0)  # Add batch dim

        # Inference
        t0 = time.time()
        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.nn.functional.softmax(logits, dim=1)
            label = torch.argmax(probs, dim=1).item()
        t1 = time.time()

        latency_ms = (t1 - t0) * 1000
        fps = 1.0 / (t1 - t0 + 1e-8)

        print(f"Inference time: {latency_ms:.2f} ms | FPS: {fps:.2f} | Persons: {label}", flush=True)

        if not args.no_window:
            # Draw label on frame and show
            disp_img = cv2.cvtColor(img_tensor.squeeze(0).numpy().transpose(1, 2, 0) * std + mean, cv2.COLOR_RGB2BGR)
            disp_img = (disp_img * 255).clip(0, 255).astype(np.uint8)
            cv2.putText(disp_img, f"Persons: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow("Webcam Inference", disp_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()