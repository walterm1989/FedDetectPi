# README de Transformación

Este directorio contiene el script de transformación que consolida métricas desde CSVs crudos bajo `Metricas_informe/raw/**`.

- Script: `transformacion.py`
- Entrada por defecto: `Metricas_informe/raw`
- Salida por defecto: `Metricas_informe/raw/metrica_consolidada.csv`

Uso:

- Ejecutar con valores por defecto:
  `python Metricas_informe/transform/transformacion.py`

- Especificar rutas:
  `python Metricas_informe/transform/transformacion.py --raw-dir Metricas_informe/raw --out Metricas_informe/raw/metrica_consolidada.csv`

- Omitir actualización automática de este README de resumen:
  `python Metricas_informe/transform/transformacion.py --no-readme`