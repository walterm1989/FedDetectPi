import os
import sys
import csv
import time
import datetime
import subprocess
import psutil

# CONFIG block
DURATION_SEC = 120
SAMPLE_INTERVAL = 2.0
SOURCE_NAME = "webcam"
METHOD = "Flower-MobileNetV3-R3"
CHILD_FLAGS = ["--no-window"]

def main():
    from pathlib import Path
    client_dir = Path(__file__).parent.resolve()
    cam_inference_path = client_dir / "cam_inference.py"

    # Prepare metrics output
    metrics_dir = Path("Metrics/raw")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = metrics_dir / f"{now}_{METHOD}_{SOURCE_NAME}.csv"
    header = ["timestamp","method","source","frame_idx","latency_ms","fps_inst","cpu_pct","ram_mb","detections"]

    # Launch inference subprocess
    cmd = [sys.executable, str(cam_inference_path)] + CHILD_FLAGS + ["--ckpt", "model_ckpt.pth"]
    proc = subprocess.Popen(cmd, cwd=client_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    psproc = psutil.Process(proc.pid)
    # warm-up
    psutil.cpu_percent(None)
    frame_idx = 0
    start_time = time.time()
    lines_written = 0

    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(header)
        while True:
            if time.time() - start_time > DURATION_SEC:
                break
            line = proc.stdout.readline()
            if not line:
                if proc.poll() is not None:
                    break
                time.sleep(0.1)
                continue
            # Parse line
            line = line.strip()
            if line.startswith("Inference time:"):
                try:
                    # Example: "Inference time: 45 ms | FPS: 22.11 | Persons: 1"
                    parts = line.split("|")
                    latency_ms = int(parts[0].split(":")[1].strip().split()[0])
                    fps_inst = float(parts[1].split(":")[1].strip())
                    detections = int(parts[2].split(":")[1].strip())
                except Exception:
                    continue
                cpu_pct = psutil.cpu_percent(interval=None)
                mem_mb = psproc.memory_info().rss / (1024*1024)
                timestamp = datetime.datetime.now().isoformat()
                writer.writerow([timestamp, METHOD, SOURCE_NAME, frame_idx, latency_ms, f"{fps_inst:.2f}", f"{cpu_pct:.1f}", f"{mem_mb:.1f}", detections])
                frame_idx += 1
                lines_written += 1
            # Sleep for sample interval only after writing (sample every N seconds)
            time.sleep(SAMPLE_INTERVAL)
    proc.terminate()
    print(f"medidor_federado: wrote {lines_written} inference lines to {csv_path}")

if __name__ == "__main__":
    main()