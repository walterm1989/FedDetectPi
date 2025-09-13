# KeyPoints: Local vs Flower

- Dataset: Metricas_informe\raw\metrica_consolidada.csv
- Filas totales (KP local+flower): 38
- Umbral P95 local (latency_ms): 30321.500

## Resumen por scope

scope,fps_mean,fps_p50,fps_p90,fps_p95,fps_p99,fps_std,fps_min,fps_max,latency_ms_mean,latency_ms_p50,latency_ms_p90,latency_ms_p95,latency_ms_p99,latency_ms_std,latency_ms_min,latency_ms_max,cpu_percent_mean,cpu_percent_p95,ram_mb_mean,ram_mb_p95,n,efficiency_fps_per_10cpu,slow_frames_vs_localP95_%
flower,0.027961538461538465,0.028,0.029,0.029,0.029,0.0009992304731449792,0.024,0.029,35808.32873076923,35592.005000000005,36781.4315,37134.4205,40970.31975,1511.6608675500615,34297.072,42215.127,100.0,100.0,762.1819230769231,784.17,26,0.0027961538461538466,100.0
local,0.5066666666666667,0.475,0.889,0.89,0.89,0.3247189927343706,0.0,0.89,28768.066666666666,28908.949999999997,30293.19,30321.5,30321.5,1441.852759683201,26094.6,30321.5,0.0,0.0,812.166015625,830.58984375,12,0.0,0.0

## Deltas (Flower - Local)

metric,Flower_minus_Local,delta_pct_vs_Local
fps_mean,-0.4787051282051282,-94.48127530364371
latency_ms_mean,7040.262064102564,24.472489394847173
cpu_percent_mean,100.0,
ram_mb_mean,-49.98409254807689,-6.154418134525338
efficiency_fps_per_10cpu,0.0027961538461538466,
slow_frames_vs_localP95_%,100.0,
