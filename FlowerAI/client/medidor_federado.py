"""
Federated metrics module for FlowerAI clients.
"""

import os
import sys
import time
import csv
import subprocess
import threading
from datetime import datetime
import psutil

# Config constants
DURATION_SEC = 120
CHILD_FLAGS = ["--no-window"]

METRICS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "Metrics", "raw")
os.makedirs(METRICS_DIR, exist_ok=True)

def parse_infer_line(line):
    # e.g., [INFER] time=... latency_ms=... fps=... persons=... probs=[...]
    if "[INFER]" not in line:
        return None
    try:
        parts = dict(token.split("=", 1) for token in line.strip().split() if "=" in token)
        latency = float(parts.get("latency_ms", "nan"))
        fps = float(parts.get("fps", "nan"))
        persons = int(parts.get("persons", "nan"))
        return latency, fps, persons
    except Exception:
        return None

def main():
    # Prepare output CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(METRICS_DIR, f"federated_metrics_{timestamp}.csv")
    header = ["timestamp", "latency_ms", "fps", "persons", "cpu_percent", "ram_mb"]

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        # Start cam_inference as a subprocess
        proc = subprocess.Popen(
            [sys.executable, "-m", "FlowerAI.client.cam_inference"] + CHILD_FLAGS,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
        )

        def monitor_proc():
            start_time = time.time()
            while time.time() - start_time < DURATION_SEC:
                line = proc.stdout.readline()
                if not line:
                    break
                metrics = parse_infer_line(line)
                if metrics:
                    latency, fps, persons = metrics
                    cpu_percent = psutil.cpu_percent(interval=0.1)
                    ram_mb = psutil.Process(proc.pid).memory_info().rss / 1024 / 1024 if psutil.pid_exists(proc.pid) else 0.0
                    row = [
                        time.strftime("%Y-%m-%d %H:%M:%S"),
                        f"{latency:.2f}",
                        f"{fps:.2f}",
                        persons,
                        f"{cpu_percent:.2f}",
                        f"{ram_mb:.2f}"
                    ]
                    writer.writerow(row)
                    f.flush()
            try:
                proc.terminate()
            except Exception:
                pass

        t = threading.Thread(target=monitor_proc)
        t.start()
        t.join(timeout=DURATION_SEC+10)
        proc.wait(timeout=5)

    # Summarize
    print(f"Federated metrics written to {csv_path}")
    # Optionally, print summary stats
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        print(df.describe())
    except Exception as e:
        print("Summary unavailable:", e)

if __name__ == "__main__":
    main()