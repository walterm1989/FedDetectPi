#!/usr/bin/env python3
import argparse
import csv
import datetime
import os
import sys
import threading
import time
import subprocess
import signal
import re

# Try to import psutil if available
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Regex for extracting inference time
INFERENCE_REGEX = re.compile(r"Inference time:\s*([0-9.]+)\s*ms", re.IGNORECASE)

def get_resource_metrics(pid=None):
    """
    Returns (cpu_percent, ram_mb) for current process (or given pid).
    If psutil is unavailable, falls back to loadavg and /proc/meminfo.
    """
    if HAS_PSUTIL:
        try:
            proc = psutil.Process(pid or os.getpid())
            cpu = proc.cpu_percent(interval=None)
            ram = proc.memory_info().rss / (1024 * 1024)
            return cpu, ram
        except Exception:
            pass  # Fallback below

    # Fallback: get system loadavg (1min) and RAM usage from /proc/meminfo
    try:
        # Loadavg: normalized to %CPU by dividing by os.cpu_count()
        load1, _, _ = os.getloadavg()
        cpu = 100.0 * load1 / (os.cpu_count() or 1)
    except Exception:
        cpu = 0.0
    try:
        with open("/proc/meminfo") as f:
            meminfo = {line.split(':')[0]: int(line.split()[1]) for line in f if ':' in line}
        ram_total = meminfo.get("MemTotal", 0)
        ram_available = meminfo.get("MemAvailable", 0)
        ram_used_kb = ram_total - ram_available
        ram = ram_used_kb / 1024.0
    except Exception:
        ram = 0.0
    return cpu, ram

def resource_sampler(interval, stop_event, latest_metrics, metrics_lock, target_pid):
    """
    Background thread: samples resource usage every interval seconds.
    Stores latest sample in latest_metrics dict (with lock).
    """
    if HAS_PSUTIL:
        # Warm up psutil for correct cpu_percent
        try:
            psutil.Process(target_pid).cpu_percent(interval=None)
        except Exception:
            pass

    while not stop_event.is_set():
        cpu, ram = get_resource_metrics(target_pid)
        with metrics_lock:
            latest_metrics['cpu_percent'] = cpu
            latest_metrics['ram_mb'] = ram
        stop_event.wait(interval)

def terminate_process(proc, timeout=5):
    """
    Gracefully terminate a subprocess: send SIGINT, then SIGTERM if needed.
    """
    try:
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait(timeout=timeout)
    except Exception:
        pass

def main():
    parser = argparse.ArgumentParser(
        description="Monitor metrics for inferencia_raspberry.py (latency, FPS, CPU, RAM) and save to CSV."
    )
    parser.add_argument('--duration', type=int, default=60, help="Duration to run (seconds, default=60)")
    parser.add_argument('--interval', type=float, default=2, help="Metrics sample interval (seconds, default=2)")
    parser.add_argument('--output', default="metricas_keypoints.csv", help="CSV file output path")
    # Parse known args, leave rest for inferencia_raspberry.py
    args, extra = parser.parse_known_args()

    # Compose command for child process
    child_cmd = [
        sys.executable, "-u", "inferencia_raspberry.py"
    ] + extra

    # Prepare CSV file
    csv_columns = ["timestamp", "latency_ms", "fps", "cpu_percent", "ram_mb"]
    try:
        csvfile = open(args.output, "w", newline="")
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(csv_columns)
        csvfile.flush()
    except Exception as e:
        print(f"Error: Could not open CSV file for writing: {e}", file=sys.stderr)
        return 1

    # Metrics state
    metrics_lock = threading.Lock()
    latest_metrics = {'cpu_percent': 0.0, 'ram_mb': 0.0}
    stop_event = threading.Event()

    # Statistics for summary
    latencies = []
    fpses = []
    cpu_usages = []
    ram_usages = []
    n_samples = 0

    # Start child process
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        child_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True,
        env=env,
    )

    # Start resource sampler thread (after proc launched)
    sampler_thread = threading.Thread(
        target=resource_sampler,
        args=(args.interval, stop_event, latest_metrics, metrics_lock, proc.pid),
        daemon=True,
    )
    sampler_thread.start()

    start_time = time.time()
    shutdown_reason = None

    try:
        for line in proc.stdout:
            print(line, end="")  # Forward to our own stdout
            now = datetime.datetime.now().isoformat()
            match = INFERENCE_REGEX.search(line)
            if match:
                try:
                    latency_ms = float(match.group(1))
                    fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0
                    with metrics_lock:
                        cpu_percent = latest_metrics.get('cpu_percent', 0.0)
                        ram_mb = latest_metrics.get('ram_mb', 0.0)
                    csvwriter.writerow([now, latency_ms, fps, cpu_percent, ram_mb])
                    csvfile.flush()
                    latencies.append(latency_ms)
                    fpses.append(fps)
                    cpu_usages.append(cpu_percent)
                    ram_usages.append(ram_mb)
                    n_samples += 1
                except Exception:
                    pass
            # Check for time-based shutdown
            if (time.time() - start_time) >= args.duration:
                shutdown_reason = "duration"
                break
    except KeyboardInterrupt:
        shutdown_reason = "ctrl-c"
    except OSError:
        shutdown_reason = "oserror"
    finally:
        stop_event.set()
        terminate_process(proc, timeout=5)
        sampler_thread.join(timeout=5)
        if proc.stdout:
            try:
                proc.stdout.close()
            except Exception:
                pass
        csvfile.close()

    # Print summary
    print("\n--- Métricas resumidas ---")
    if n_samples:
        mean_latency = sum(latencies) / n_samples
        mean_fps = sum(fpses) / n_samples
        mean_cpu = sum(cpu_usages) / n_samples
        mean_ram = sum(ram_usages) / n_samples
        print(f"Duración: {round(time.time() - start_time, 2)}s ({'Ctrl-C' if shutdown_reason == 'ctrl-c' else 'Tiempo cumplido'})")
        print(f"Número de muestras: {n_samples}")
        print(f"Latencia promedio (ms): {mean_latency:.2f}")
        print(f"FPS promedio: {mean_fps:.2f}")
        print(f"CPU promedio (%): {mean_cpu:.2f}")
        print(f"RAM promedio (MB): {mean_ram:.2f}")
    else:
        print("No se recolectaron muestras de inferencia.")
    return 0

if __name__ == "__main__":
    sys.exit(main())