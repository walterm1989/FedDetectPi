#!/usr/bin/env python3
import csv
import datetime
import os
import sys
import threading
import time
import subprocess
import signal
import re
from pathlib import Path

# === CONFIGURACIÓN LOCAL (editar aquí si hace falta) ===
DURATION_SEC      = 240        # duración total de la medición
SAMPLE_INTERVAL   = 2.0        # cada cuántos segundos se muestrea CPU/RAM
SOURCE_NAME       = "webcam"   # etiqueta que irá al CSV y al nombre del archivo
METHOD            = "KeyPoints-resnet50"  # cambia a 'KeyPoints-mobile' si usas MobileNet; puedes añadir sufijos '-TS' y/o '-half'
CHILD_FLAGS       = ["--draw-box"]  # flags opcionales para inferencia_raspberry.py; dejar [] si no quieres pasar ninguno

# Try to import psutil if available
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Regex for extracting inference time, FPS and Persons (detections)
INFERENCE_REGEX = re.compile(r"Inference time:\s*([0-9.]+)\s*ms", re.IGNORECASE)
FPS_REGEX = re.compile(r"FPS:\s*([0-9.]+)", re.IGNORECASE)
PERSONS_REGEX = re.compile(r"Persons:\s*([0-9]+)", re.IGNORECASE)

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
    # Set up child process command and working directory
    script_dir = Path(__file__).resolve().parent
    child_script = script_dir / "inferencia_raspberry.py"
    child_cmd = [sys.executable, "-u", str(child_script)] + list(CHILD_FLAGS)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    # Prepare CSV output path: Metrics/raw/YYYYmmdd_HHMMSS_<METHOD>_<SOURCE_NAME>.csv
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_method = re.sub(r'[^a-zA-Z0-9_-]', '', METHOD)
    safe_source = re.sub(r'[^a-zA-Z0-9_-]', '', SOURCE_NAME)
    repo_root = script_dir.parent
    csv_dir = repo_root / "Metrics" / "raw"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / f"{now_str}_{safe_method}_{safe_source}.csv"

    csv_columns = [
        "timestamp", "method", "source", "frame_idx",
        "latency_ms", "fps_inst", "cpu_pct", "ram_mb", "detections"
    ]
    try:
        csvfile = open(csv_path, "w", newline="")
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
    try:
        proc = subprocess.Popen(
            child_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
            env=env,
            cwd=str(script_dir),
        )
    except FileNotFoundError:
        print(f"\n[ERROR] No se encontró el script hijo: {child_script.resolve()}\n"
              f"Por favor verifica que el archivo existe en la ruta indicada.", file=sys.stderr)
        return 1

    # Warm-up psutil cpu_percent for child process
    if HAS_PSUTIL:
        try:
            psutil.Process(proc.pid).cpu_percent(interval=None)
        except Exception:
            pass

    # Start resource sampler thread (after proc launched)
    sampler_thread = threading.Thread(
        target=resource_sampler,
        args=(SAMPLE_INTERVAL, stop_event, latest_metrics, metrics_lock, proc.pid),
        daemon=True,
    )
    sampler_thread.start()

    start_time = time.time()
    shutdown_reason = None
    frame_idx = 0

    try:
        for line in proc.stdout:
            print(line, end="")  # Forward to our own stdout
            timestamp = datetime.datetime.now().isoformat()
            inf_match = INFERENCE_REGEX.search(line)
            if inf_match:
                try:
                    latency_ms = float(inf_match.group(1))
                except Exception:
                    latency_ms = None
                # Try to get FPS from output, else compute
                fps_match = FPS_REGEX.search(line)
                if fps_match:
                    try:
                        fps_inst = float(fps_match.group(1))
                    except Exception:
                        fps_inst = 1000.0 / latency_ms if latency_ms and latency_ms > 0 else 0.0
                else:
                    fps_inst = 1000.0 / latency_ms if latency_ms and latency_ms > 0 else 0.0

                # Try to get detections/persons from output
                persons_match = PERSONS_REGEX.search(line)
                detections = int(persons_match.group(1)) if persons_match else ""

                # Synchronously fetch CPU/RAM for the child process right before writing
                if HAS_PSUTIL:
                    try:
                        child_proc = psutil.Process(proc.pid)
                        cpu_pct = child_proc.cpu_percent(interval=None)
                        ram_mb = child_proc.memory_info().rss / (1024 * 1024)
                    except Exception:
                        cpu_pct, ram_mb = 0.0, 0.0
                else:
                    # Fallback for no psutil: use system load/mem
                    cpu_pct, ram_mb = get_resource_metrics(proc.pid)

                csvwriter.writerow([
                    timestamp, METHOD, SOURCE_NAME, frame_idx,
                    latency_ms, fps_inst, cpu_pct, ram_mb, detections
                ])
                csvfile.flush()
                latencies.append(latency_ms if latency_ms is not None else 0.0)
                fpses.append(fps_inst)
                cpu_usages.append(cpu_pct)
                ram_usages.append(ram_mb)
                n_samples += 1
                frame_idx += 1
            # Check for time-based shutdown
            if (time.time() - start_time) >= DURATION_SEC:
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
