# Sección 4.2: Consumo de Recursos

Este análisis contiene métricas de consumo de CPU y RAM para cada método evaluado en el edge.
Se incluyen medias, percentiles, máximos, márgenes de memoria y notas de uso recomendadas.

## Figuras
- [Resumen CPU (barras)](figs\fig_42_resumen_cpu_barras.png)
- [Resumen RAM (barras)](figs\fig_42_resumen_ram_barras.png)
- [Boxplot CPU (%)](figs\fig_42_box_cpu_pct.png)
- [Boxplot RAM (MB)](figs\fig_42_box_ram_mb.png)
- [Histograma CPU (%)](figs\fig_42_hist_cpu_pct.png)
- [Histograma RAM (MB)](figs\fig_42_hist_ram_mb.png)

## Tablas
- [Tabla resumen CSV](tables\tabla_42_recursos_local.csv)
- [Tabla resumen Markdown](tables\tabla_42_recursos_local.md)

## Notas sobre umbrales y recomendaciones
- CPU media &gt;80% o P95 &gt;90%: Mejorar eficiencia.
- Margen libre RAM &lt;512MB: Riesgo, puede requerir optimización.

## Detalle de métodos

| Método | Nota |
|---|---|
| KeyPoints-ResNet50 | OK |
| BBoxes-YOLOv4tiny | Mejorar: CPU media >80%, CPU P95 >90% |
| FlowerAI | Mejorar: CPU media >80%, CPU P95 >90% |
