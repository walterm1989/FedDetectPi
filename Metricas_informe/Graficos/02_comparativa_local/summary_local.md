# Comparativa LOCAL (BoundingBoxes vs KeyPoints)

- Dataset: Metricas_informe\raw\metrica_consolidada.csv
- Filas totales (LOCAL): 219

## Resumen por método

method,fps_mean,fps_p50,fps_p90,fps_p95,fps_p99,fps_std,fps_min,fps_max,latency_ms_mean,latency_ms_p50,latency_ms_p90,latency_ms_p95,latency_ms_p99,latency_ms_std,latency_ms_min,latency_ms_max,cpu_percent_mean,cpu_percent_p95,ram_mb_mean,ram_mb_p95,n,efficiency_fps_per_10cpu
BoundingBoxes,2.3413526570048306,2.36,2.39,2.4,2.41,0.10560094457595177,0.94,2.42,429.10743961352654,423.74,437.174,440.815,444.0524,45.04326357956352,413.89,1065.55,99.51690821256038,100.0,308.58164251207734,311.0,207,0.23527184466019416
KeyPoints,0.5066666666666667,0.475,0.889,0.89,0.89,0.3247189927343706,0.0,0.89,28768.066666666666,28908.949999999997,30293.19,30321.5,30321.5,1441.852759683201,26094.6,30321.5,0.0,0.0,812.166015625,830.58984375,12,0.0

## Deltas (KeyPoints - BoundingBoxes)

metric,KP_minus_BB,delta_pct_vs_BB
fps_mean,-1.8346859903381638,-78.3600874840094
latency_ms_mean,28338.95922705314,6604.164041662032
cpu_percent_mean,-99.51690821256038,-100.0
ram_mb_mean,503.58437311292266,163.19323761886233
efficiency_fps_per_10cpu,-0.23527184466019416,-100.0
