import os
import sys
import subprocess
import psutil
import csv
import time
import datetime
import select

# ----- CONFIG -----
DURATION_SEC    = 120
SAMPLE_INTERVAL = 2.0
SOURCE_NAME     = "webcam"
METHOD          = "Flower-MobileNetV3-R3"
CHILD_FLAGS     = ["--no-window"]

# Paths
DIR_CLIENT = os.path.dirname(os.path.abspath(__file__))
DIR_METRICS = os.path.join(DIR_CLIENT, "..", "Metrics", "raw")
CAM_INFERENCE = os.path.join(DIR_CLIENT, "cam_inference.py")

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def csv_filename():
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{now}_{METHOD}_{SOURCE_NAME}.csv"
    return os.path.join(DIR_METRICS, name)

def parse_cam_line(line):
    """
    Parse a line from cam_inference.py's output.
    Expected example format (adapt as needed):
    frame_idx,latency_ms,fps_inst,persons
    or
    Frame 17: latency=42ms, fps=22.3, persons=1
    Returns (frame_idx, latency_ms, fps_inst, detections) or None if cannot parse.
    """
    # Try CSV-style parsing first
    try:
        parts = [x.strip() for x in line.strip().split(",")]
        if len(parts) == 4:
            frame_idx = int(parts[0])
            latency_ms = float(parts[1])
            fps_inst = float(parts[2])
            detections = int(parts[3])
            return frame_idx, latency_ms, fps_inst, detections
    except Exception:
        pass
    # Try key=val parsing
    try:
        # e.g. "Frame 17: latency=42ms, fps=22.3, persons=1"
        import re
        m = re.search(r"Frame\s*(\d+):.*latency=([\d\.]+)ms.*fps=([\d\.]+).*persons=(\d+)", line)
        if m:
            frame_idx = int(m.group(1))
            latency_ms = float(m.group(2))
            fps_inst = float(m.group(3))
            detections = int(m.group(4))
            return frame_idx, latency_ms, fps_inst, detections
    except Exception:
        pass
    return None

def main():
    ensure_dir(DIR_METRICS)
    out_csv = csv_filename()
    start_time = time.time()
    end_time = start_time + DURATION_SEC

    # Spawn cam_inference.py subprocess
    cmd = [sys.executable, os.path.abspath(CAM_INFERENCE)] + CHILD_FLAGS
    proc = subprocess.Popen(
        cmd,
        cwd=DIR_CLIENT,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=1,
        universal_newlines=True
    )
    # Wrap with psutil
    p_proc = psutil.Process(proc.pid)
    p_proc.cpu_percent(None)  # warm up

    # Open CSV
    with open(out_csv, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        header = [
            "timestamp", "method", "source", "frame_idx",
            "latency_ms", "fps_inst", "cpu_pct", "ram_mb", "detections"
        ]
        writer.writerow(header)

        frame_count = 0

        try:
            while True:
                now = time.time()
                if now >= end_time:
                    break
                # select for stdout with timeout
                timeout = max(0, min(SAMPLE_INTERVAL, end_time - now))
                rlist, _, _ = select.select([proc.stdout], [], [], timeout)
                if rlist:
                    line = proc.stdout.readline()
                    if not line:
                        # Process ended
                        break
                    parsed = parse_cam_line(line)
                    if parsed is not None:
                        frame_idx, latency_ms, fps_inst, detections = parsed
                        cpu_pct = p_proc.cpu_percent(interval=None)
                        ram_mb = p_proc.memory_info().rss / (1024 * 1024)
                        timestamp = datetime.datetime.now().isoformat()
                        writer.writerow([
                            timestamp, METHOD, SOURCE_NAME, frame_idx,
                            latency_ms, fps_inst, cpu_pct, ram_mb, detections
                        ])
                        frame_count += 1
                else:
                    # No output this interval; continue
                    continue

        except KeyboardInterrupt:
            print("Interrupted by user. Terminating...")
        finally:
            # Cleanup
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
            print(f"Summary: {frame_count} frames processed.")
            print(f"Results saved to: {out_csv}")

if __name__ == "__main__":
    main()