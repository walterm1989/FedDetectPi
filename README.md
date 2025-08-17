# FedDetectPi
Sistema de detección de personas basado en aprendizaje federado con visión artificial distribuida en Raspberry Pi y portátil usando Flower AI.

---

## Raspberry Pi defaults

`inferencia_raspberry.py` is pre-configured with optimized defaults for running real-time keypoint detection on a Raspberry Pi 4. The main constants are defined at the top of the script:

- `DEFAULT_CAP_WIDTH` / `DEFAULT_CAP_HEIGHT`: Camera capture resolution (default: 320x240).
- `DEFAULT_PROC_SCALE`: Factor to downscale frames before inference (default: 0.5). Lower for faster inference, higher for more detail.
- `DEFAULT_SKIP_FRAMES`: Number of frames to skip between inferences (default: 2, i.e., run inference every 3rd frame).
- `DEFAULT_HALF_PRECISION`: Whether to use float16 ("half precision") on supported hardware (default: True).
- `TARGET_LATENCY_MS`: Target latency per inference call, in milliseconds (default: 200 ms). The script adaptively adjusts `skip_frames` to try to meet this.

You can override these defaults via command-line arguments. For example:
```
python3 inferencia_raspberry.py --width 640 --height 480 --proc-scale 0.8 --skip-frames 0 --no-half --target-latency-ms 300
```

To force or disable half-precision, use `--half` or `--no-half`.

Tweak these parameters to balance detection quality and real-time speed, depending on your Raspberry Pi's performance and workload.

---

## KeyPoints

Para ejecutar el medidor de métricas con los valores por defecto, usa:
```
python KeyPoints/medidor_metricas.py
```

Para cambiar duración, fuente o flags del hijo, edita el bloque CONFIG al inicio de `medidor_metricas.py`.

Esto lanzará `inferencia_raspberry.py` como proceso hijo, recolectará métricas de inferencia, CPU y RAM, y guardará los resultados en un archivo CSV en `Metrics/raw/`.

```
