import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tabulate import tabulate

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Cobertura de Detección - Sección 4.3")
    parser.add_argument('--input', default="Metrics/out/section4_metrics_all.csv", help="Input CSV file")
    parser.add_argument('--out', default="Metrics/4.3_Capacidad_Deteccion/figs", help="Output figures directory")
    parser.add_argument('--tables', default="Metrics/4.3_Capacidad_Deteccion/tables", help="Output tables directory")
    parser.add_argument('--format', choices=['png', 'pdf'], default="png", help="Figure format")
    parser.add_argument('--dpi', type=int, default=200, help="Figure DPI")
    parser.add_argument('--only', default=None, help="Comma-separated list of methods to include")
    parser.add_argument('--min-detections', type=int, default=1, help="Minimum detections to count as covered")
    parser.add_argument('--presence', default=None, help="Optional presence file (frames, intervals, seconds)")
    parser.add_argument('--presence-mode', choices=['frames', 'intervals', 'seconds'], default=None, help="How to interpret presence file")
    parser.add_argument('--window-frames', type=int, default=10, help="Rolling window for smoothing in timeline plots")
    parser.add_argument('--make-readme', action='store_true', help="Generate README_4.3.md with figures/tables summary")
    return parser.parse_args()

def load_data(input_csv, only_methods=None):
    df = pd.read_csv(input_csv)
    df.columns = df.columns.str.strip()
    # Convert numerics except for method column
    for c in df.columns:
        if c not in ("method", "frame_id", "elapsed_sec"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if only_methods is not None:
        only = [m.strip() for m in only_methods.split(",") if m.strip()]
        df = df[df["method"].isin(only)].reset_index(drop=True)
    return df

def load_presence(presence_path, mode, df):
    # Returns: Series mask (index aligned with df) (bool: True if present), and list of (start,end) intervals in frames or seconds
    if presence_path is None or mode is None:
        N = len(df)
        return pd.Series([True]*N, index=df.index), []
    presence = pd.read_csv(presence_path)
    if mode == "frames":
        # Must have columns: frame_id, present (0/1)
        presence.columns = [c.strip() for c in presence.columns]
        mask = pd.Series(0, index=df.index)
        presence_map = dict(zip(presence['frame_id'], presence['present']))
        mask = df['frame_id'].map(presence_map).fillna(0).astype(bool)
        intervals = []
    elif mode == "intervals":
        # Columns: start_frame, end_frame
        mask = pd.Series(False, index=df.index)
        intervals = []
        for _, row in presence.iterrows():
            startf, endf = int(row['start_frame']), int(row['end_frame'])
            in_interval = (df['frame_id'] >= startf) & (df['frame_id'] <= endf)
            mask = mask | in_interval
            intervals.append((startf, endf))
    elif mode == "seconds":
        # Columns: start_sec, end_sec
        mask = pd.Series(False, index=df.index)
        intervals = []
        for _, row in presence.iterrows():
            starts, ends = float(row['start_sec']), float(row['end_sec'])
            in_interval = (df['elapsed_sec'] >= starts) & (df['elapsed_sec'] <= ends)
            mask = mask | in_interval
            intervals.append((starts, ends))
    else:
        raise ValueError(f"Unknown presence mode: {mode}")
    return mask, intervals

def compute_aggregates(df, presence_mask, intervals, min_detections=1):
    results = []
    methods = sorted(df['method'].unique())
    for method in methods:
        d = df[df['method'] == method].copy()
        pres = presence_mask.loc[d.index]
        # Coverage
        denom = pres.sum() if pres.any() else len(d)
        num = ((d['detections'] >= min_detections) & pres).sum() if pres.any() else (d['detections'] >= min_detections).sum()
        pct_coverage = 100 * num / denom if denom > 0 else np.nan
        # Detections/frame mean
        det_per_frame = d['detections'][pres].mean() if pres.any() else d['detections'].mean()
        # Gap max
        # Only within presence
        mask = pres.values
        gap_max = 0
        curr_gap = 0
        for det, present in zip(d['detections'], mask):
            if present:
                if det >= min_detections:
                    gap_max = max(gap_max, curr_gap)
                    curr_gap = 0
                else:
                    curr_gap += 1
        gap_max = max(gap_max, curr_gap)
        # t_median to first detection (ms)
        t_medians = []
        # Only if intervals and elapsed_sec exist
        if intervals and 'elapsed_sec' in d.columns:
            for (start, end) in intervals:
                if isinstance(start, int):
                    # frames
                    rows = d[(d['frame_id'] >= start) & (d['frame_id'] <= end)]
                else:
                    # seconds
                    rows = d[(d['elapsed_sec'] >= start) & (d['elapsed_sec'] <= end)]
                first = rows[rows['detections'] >= min_detections]
                if not rows.empty and not first.empty:
                    t_first = first.iloc[0]['elapsed_sec'] - rows.iloc[0]['elapsed_sec']
                    t_medians.append(t_first * 1000.0)  # ms
            t_median = np.median(t_medians) if t_medians else np.nan
        else:
            t_median = np.nan
        # False positives
        if presence_mask.any():
            fp_mask = ~presence_mask
            n_fp = ((d['detections'] >= min_detections) & fp_mask.loc[d.index]).sum()
            denom_fp = fp_mask.loc[d.index].sum()
            false_pos = 100 * n_fp / denom_fp if denom_fp > 0 else np.nan
        else:
            false_pos = np.nan
        results.append(dict(
            method=method,
            pct_coverage=np.round(pct_coverage, 1),
            det_per_frame=np.round(det_per_frame, 3) if not np.isnan(det_per_frame) else np.nan,
            gap_max=int(gap_max),
            t_median=np.round(t_median, 1) if not np.isnan(t_median) else np.nan,
            false_pos=np.round(false_pos, 1) if not np.isnan(false_pos) else np.nan,
        ))
    return results

def make_note_coverage(row):
    """
    Generates a tiered, positive-phrased coverage note with method-specific suggestions.
    Tiers and suggestions follow the project spec.
    """
    coverage = row['% cobertura']
    gap = row['Gap máx sin detección (frames)']
    method = row['method']

    # Tier logic
    if (coverage >= 85) and (gap <= 10):
        note = ("Adecuado para presencia/aforo en edge; mantener resolución/umbral y considerar ROI si hay oclusiones.")
        tier = "A"
    elif (60 <= coverage < 85) or (11 <= gap <= 30):
        note = ("Operable para monitorización por eventos; mejorar con ajuste de threshold/NMS, ROI sobre zona de interés y resolución 320×240.")
        tier = "B"
    elif (coverage < 60) or (gap > 30):
        note = ("Señal complementaria / análisis offline; considerar modelos más ligeros o flujo por etapas (caja→pose) y optimizar umbral/ROI.")
        tier = "C"
    else:  # Safety fallback
        note = "Cobertura evaluada; revisar detalles según necesidades específicas."
        tier = "?"

    # Method-specific suggestions
    suggestions = {
        "KeyPoints-ResNet50": " Explorar variantes lightweight (MoveNet/BlazePose) o caja→pose.",
        "BBoxes-YOLOv4tiny": " Ajustar umbral/NMS y definir ROI; reducir resolución si hay oclusiones.",
        "FlowerAI": " Sincronizar ciclos de captura/proceso y tasa de muestreo; evaluar ventanas de suavizado.",
    }
    if method in suggestions:
        note += suggestions[method]

    return note

def save_tables(results, out_dir):
    cols = [
        "method", "% cobertura", "Detecciones/frame (media)", "Gap máx sin detección (frames)",
        "T_mediana a 1ª detección (ms)", "Falsos positivos (%)", "Aplicabilidad práctica (y mejoras)"
    ]
    # Prepare DataFrame
    df = pd.DataFrame(results)
    df["% cobertura"] = df["pct_coverage"]
    df["Detecciones/frame (media)"] = df["det_per_frame"]
    df["Gap máx sin detección (frames)"] = df["gap_max"]
    df["T_mediana a 1ª detección (ms)"] = df["t_median"]
    df["Falsos positivos (%)"] = df["false_pos"]
    # Notes
    df["Aplicabilidad práctica (y mejoras)"] = df.apply(lambda row: make_note_coverage({
        "method": row["method"],
        "% cobertura": row["% cobertura"],
        "Gap máx sin detección (frames)": row["Gap máx sin detección (frames)"]
    }), axis=1)
    df = df[cols]
    # Sort
    df = df.sort_values(by=["% cobertura", "Gap máx sin detección (frames)", "method"], ascending=[False, True, True])
    csv_path = os.path.join(out_dir, "tabla_43_cobertura_local.csv")
    md_path = os.path.join(out_dir, "tabla_43_cobertura_local.md")
    df.to_csv(csv_path, index=False)
    with open(md_path, "w", encoding="utf8") as f:
        f.write(tabulate(df, headers=cols, tablefmt="github", showindex=False))
    return df

def plot_bars_coverage(df, out_dir, fmt="png", dpi=200):
    plt.figure(figsize=(8, 4))
    bars = plt.bar(df['method'], df["% cobertura"], color='skyblue')
    plt.ylabel("% cobertura")
    plt.title("Cobertura por método")
    plt.grid(axis='y', linestyle=':')
    # Annotate
    for bar in bars:
        plt.annotate(f'{bar.get_height():.1f}%', (bar.get_x() + bar.get_width()/2, bar.get_height()), 
                     ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    fn = f"fig_43_resumen_cobertura_barras.{fmt}"
    plt.savefig(os.path.join(out_dir, fn), dpi=dpi)
    plt.close()

def plot_bars_detpf(df, out_dir, fmt="png", dpi=200):
    plt.figure(figsize=(8, 4))
    bars = plt.bar(df['method'], df["Detecciones/frame (media)"], color='mediumseagreen')
    plt.ylabel("Detecciones/frame (media)")
    plt.title("Detecciones por frame por método")
    plt.grid(axis='y', linestyle=':')
    for bar in bars:
        plt.annotate(f'{bar.get_height():.2f}', (bar.get_x() + bar.get_width()/2, bar.get_height()), 
                     ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    fn = f"fig_43_resumen_det_per_frame.{fmt}"
    plt.savefig(os.path.join(out_dir, fn), dpi=dpi)
    plt.close()

def plot_hist_detections(df, out_dir, fmt="png", dpi=200):
    plt.figure(figsize=(8, 4))
    for method in df['method'].unique():
        d = df[df['method'] == method]
        plt.hist(d['detections'], bins=range(0, d['detections'].max()+2), alpha=0.5, label=method)
    plt.xlabel("Detecciones por frame")
    plt.ylabel("Frames")
    plt.title("Histograma de detecciones por frame")
    plt.legend()
    plt.tight_layout()
    fn = f"fig_43_hist_detecciones.{fmt}"
    plt.savefig(os.path.join(out_dir, fn), dpi=dpi)
    plt.close()

def plot_timelines(df, out_dir, fmt="png", dpi=200, window_frames=10, presence_mask=None):
    for method in df['method'].unique():
        d = df[df['method'] == method].sort_values('frame_id')
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(d['frame_id'], d['detections'], label="Detecciones", alpha=0.6)
        # Smoothed
        if len(d['detections']) >= window_frames:
            smooth = d['detections'].rolling(window_frames, center=True, min_periods=1).mean()
            ax.plot(d['frame_id'], smooth, label=f"Suavizado ({window_frames} frames)", color='orange', linewidth=2)
        # Presence overlay
        if presence_mask is not None:
            pres = presence_mask.loc[d.index]
            ax.fill_between(d['frame_id'], 0, d['detections'].max(), where=pres, color='lightgreen', alpha=0.2, label="Presencia")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Detecciones")
        ax.set_title(f"Timeline de detecciones - {method}")
        ax.grid(linestyle=':')
        ax.legend()
        plt.tight_layout()
        fn = f"fig_43_timeline_detecciones_{method}.{fmt}"
        plt.savefig(os.path.join(out_dir, fn), dpi=dpi)
        plt.close()

def plot_gaps_cdf(df, out_dir, fmt="png", dpi=200, min_detections=1, presence_mask=None):
    # Optional: CDF of gaps between detections per method
    for method in df['method'].unique():
        d = df[df['method'] == method]
        mask = presence_mask.loc[d.index] if presence_mask is not None else np.ones(len(d), bool)
        gaps = []
        curr_gap = 0
        for det, present in zip(d['detections'], mask):
            if present:
                if det >= min_detections:
                    if curr_gap > 0:
                        gaps.append(curr_gap)
                    curr_gap = 0
                else:
                    curr_gap += 1
        if curr_gap > 0:
            gaps.append(curr_gap)
        if gaps:
            sorted_gaps = np.sort(gaps)
            cdf = np.arange(1, len(sorted_gaps)+1) / len(sorted_gaps)
            plt.figure(figsize=(7,4))
            plt.plot(sorted_gaps, cdf, marker='o')
            plt.xlabel("Gap sin detección (frames)")
            plt.ylabel("CDF")
            plt.title(f"CDF de huecos sin detección - {method}")
            plt.grid(linestyle=':')
            plt.tight_layout()
            fn = f"fig_43_gaps_cdf_{method}.{fmt}"
            plt.savefig(os.path.join(out_dir, fn), dpi=dpi)
            plt.close()

def write_readme(fig_dir, table_dir, dftab, formats, presence_args):
    readme_path = os.path.join(os.path.dirname(fig_dir), "README_4.3.md")
    lines = []
    lines.append("# Sección 4.3: Capacidad de Detección\n")
    lines.append("Este análisis resume la cobertura y el desempeño de los métodos de detección evaluados.\n")
    lines.append("## Figuras generadas\n")
    for fig in [
        "fig_43_resumen_cobertura_barras",
        "fig_43_resumen_det_per_frame",
        "fig_43_hist_detecciones",
    ]:
        for fmt in formats:
            lines.append(f"- ![Resumen {fig}]({os.path.relpath(os.path.join(fig_dir, f'{fig}.{fmt}'), os.path.dirname(readme_path))})")
    for method in dftab['method']:
        for fmt in formats:
            lines.append(f"- ![Timeline {method}]({os.path.relpath(os.path.join(fig_dir, f'fig_43_timeline_detecciones_{method}.{fmt}'), os.path.dirname(readme_path))})")
    # Tables
    for fn in ["tabla_43_cobertura_local.csv", "tabla_43_cobertura_local.md"]:
        lines.append(f"- [{fn}]({os.path.relpath(os.path.join(table_dir, fn), os.path.dirname(readme_path))})")
    if presence_args:
        lines.append(f"\n> Nota: Se utilizó archivo de presencia ({presence_args['presence']}) en modo {presence_args['presence_mode']}.")
    else:
        lines.append("\n> Nota: El análisis se realizó considerando todos los frames como presencia.")
    with open(readme_path, "w", encoding="utf8") as f:
        f.write("\n".join(lines))

def main():
    args = parse_args()
    ensure_dir(args.out)
    ensure_dir(args.tables)

    print(f"[INFO] Leyendo datos de: {args.input}")
    df = load_data(args.input, only_methods=args.only)
    if df.empty:
        print("[ERROR] No se encontraron datos para procesar.")
        return

    print("[INFO] Procesando presencia" + (f" desde {args.presence} (modo {args.presence_mode})" if args.presence else " (sin presencia explícita)"))
    presence_mask, intervals = load_presence(args.presence, args.presence_mode, df)
    if presence_mask is None:
        presence_mask = pd.Series([True]*len(df), index=df.index)

    print("[INFO] Calculando métricas agregadas por método...")
    results = compute_aggregates(df, presence_mask, intervals, min_detections=args.min_detections)

    print("[INFO] Guardando tablas...")
    dftab = save_tables(results, args.tables)

    print("[INFO] Generando figuras resumen...")
    plot_bars_coverage(dftab, args.out, fmt=args.format, dpi=args.dpi)
    plot_bars_detpf(dftab, args.out, fmt=args.format, dpi=args.dpi)
    plot_hist_detections(df, args.out, fmt=args.format, dpi=args.dpi)
    print("[INFO] Generando timelines...")
    plot_timelines(df, args.out, fmt=args.format, dpi=args.dpi, window_frames=args.window_frames, presence_mask=presence_mask)
    # Optional CDF
    # print("[INFO] Generando CDF de gaps (opcional)...")
    # plot_gaps_cdf(df, args.out, fmt=args.format, dpi=args.dpi, min_detections=args.min_detections, presence_mask=presence_mask)

    if args.make_readme:
        print("[INFO] Escribiendo README_4.3.md...")
        write_readme(args.out, args.tables, dftab, [args.format], 
            dict(presence=args.presence, presence_mode=args.presence_mode) if args.presence else None)

    print("[INFO] Listo. Resultados guardados en:")
    print("  Figuras:", args.out)
    print("  Tablas:", args.tables)
    if args.make_readme:
        print("  README:", os.path.join(os.path.dirname(args.out), "README_4.3.md"))

if __name__ == "__main__":
    main()