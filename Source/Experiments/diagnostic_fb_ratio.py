"""
diagnostic_fb_ratio.py
Diagnostic al distributiei raportului Forward/Backward (F/B), inainte
de a-l folosi ca parametru de clasificare -- raspunde direct la
observatia profesorului: F/B poate avea o distributie foarte larga/
instabila atunci cand intensitatea bSHG (numitorul) e mica pentru unele
imagini, ceea ce ar putea produce valori extreme (outlieri) care sa
distorsioneze orice analiza ulterioara.

Analiza facuta:
    1. Statistici descriptive complete (mean, median, std, min, max,
       percentile) pe F/B brut si pe log(F/B) (transformarea log e
       standard pentru rapoarte, pentru ca simetrizeaza distributia --
       un raport de 0.1 si unul de 10 sunt "la fel de departe" de 1
       doar in scara logaritmica).
    2. Histograma F/B brut SI log(F/B), una langa alta.
    3. Scatter plot F/B vs intensitate_backward -- verifica vizual
       daca valorile extreme de F/B coincid cu intensitate backward
       foarte mica (exact ipoteza profesorului).
    4. Listarea explicita a top 10 cele mai extreme perechi (probe,
       sectiune, polarizare), cu valorile lor de intensitate, ca sa
       poti verifica manual daca sunt imagini valide sau artefacte.

Utilizare:
    python diagnostic_fb_ratio.py --fb-csv "...\\features_fb_ratio.csv"
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))


def _find_and_add_core(start: Path) -> bool:
    current = start
    for _ in range(5):
        candidate = current / "core"
        if (candidate / "config.py").exists():
            sys.path.insert(0, str(candidate))
            return True
        current = current.parent
    return False

_find_and_add_core(Path(__file__).resolve().parent)

DEFAULT_FB_CSV    = None
DEFAULT_PLOTS_DIR = None
try:
    from config import CSV_DIR, PLOTS_DIR as CONFIG_PLOTS_DIR
    DEFAULT_FB_CSV    = CSV_DIR / "features_fb_ratio.csv"
    DEFAULT_PLOTS_DIR = CONFIG_PLOTS_DIR / "fb_ratio"
except ImportError:
    pass


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnostic distributie raport F/B")
    parser.add_argument("--fb-csv", type=Path, default=DEFAULT_FB_CSV,
                         required=DEFAULT_FB_CSV is None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PLOTS_DIR)
    parser.add_argument("--top-n-outliers", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_dir = args.output_dir or (args.fb_csv.parent.parent / "Plots" / "fb_ratio")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.fb_csv.exists():
        print(f"[EROARE] Nu gasesc {args.fb_csv}. Ruleaza intai features_fb_ratio.py")
        sys.exit(1)

    df = pd.read_csv(args.fb_csv)
    print(f"\n{'='*60}")
    print(f"  Diagnostic distributie F/B")
    print(f"{'='*60}")
    print(f"  Total perechi: {len(df)}")

    fb = df["FB_ratio"].astype(float)
    log_fb = np.log(fb.replace(0, np.nan)).dropna()

    # ──────────────────────────── STATISTICI ─────────────────────────────

    print(f"\n  --- F/B brut ---")
    print(f"  Mean   : {fb.mean():.3f}")
    print(f"  Median : {fb.median():.3f}")
    print(f"  Std    : {fb.std():.3f}")
    print(f"  Min    : {fb.min():.3f}")
    print(f"  Max    : {fb.max():.3f}")
    print(f"  P5 / P95   : {fb.quantile(0.05):.3f} / {fb.quantile(0.95):.3f}")
    print(f"  P1 / P99   : {fb.quantile(0.01):.3f} / {fb.quantile(0.99):.3f}")

    print(f"\n  --- log(F/B) ---")
    print(f"  Mean   : {log_fb.mean():.3f}")
    print(f"  Std    : {log_fb.std():.3f}")
    print(f"  Min    : {log_fb.min():.3f}  (echivalent F/B = {np.exp(log_fb.min()):.4f})")
    print(f"  Max    : {log_fb.max():.3f}  (echivalent F/B = {np.exp(log_fb.max()):.2f})")

    # Detectare outlieri prin regula IQR (pe F/B brut)
    q1, q3 = fb.quantile(0.25), fb.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    n_outliers = ((fb < lower_bound) | (fb > upper_bound)).sum()
    pct_outliers = n_outliers / len(fb) * 100

    print(f"\n  --- Outlieri (regula IQR standard, 1.5x) ---")
    print(f"  Interval normal: [{lower_bound:.3f}, {upper_bound:.3f}]")
    print(f"  Outlieri detectati: {n_outliers} din {len(fb)} ({pct_outliers:.1f}%)")

    # Corelatie F/B vs intensitate backward (ipoteza profesorului)
    corr = df["FB_ratio"].corr(df["intensity_back"])
    corr_abs = df["FB_ratio"].corr(1 / df["intensity_back"].replace(0, np.nan))
    print(f"\n  --- Legatura F/B <-> intensitate backward ---")
    print(f"  Corelatie Pearson(F/B, intensitate_back)      : {corr:.3f}")
    print(f"  Corelatie Pearson(F/B, 1/intensitate_back)    : {corr_abs:.3f}")
    print(f"  (o corelatie puternic pozitiva cu 1/intensitate_back confirma")
    print(f"   ipoteza: F/B creste artificial cand backward e slab)")

    # ──────────────────────────── TOP OUTLIERI ────────────────────────────

    df_sorted = df.reindex(df["FB_ratio"].sort_values(ascending=False).index)
    print(f"\n  --- Top {args.top_n_outliers} valori F/B cele mai MARI ---")
    print(f"  {'Probe':<12} {'Sect':<6} {'Polar':<8} {'FB_ratio':>10} {'Int.back':>14} {'Int.forward':>14}")
    print(f"  {'-'*68}")
    for _, row in df_sorted.head(args.top_n_outliers).iterrows():
        print(f"  {row['probe']:<12} {row['section']:<6} {row['polar']:<8} "
              f"{row['FB_ratio']:>10.3f} {row['intensity_back']:>14.0f} {row['intensity_forward']:>14.0f}")

    df_sorted_low = df.reindex(df["FB_ratio"].sort_values(ascending=True).index)
    print(f"\n  --- Top {args.top_n_outliers} valori F/B cele mai MICI ---")
    print(f"  {'Probe':<12} {'Sect':<6} {'Polar':<8} {'FB_ratio':>10} {'Int.back':>14} {'Int.forward':>14}")
    print(f"  {'-'*68}")
    for _, row in df_sorted_low.head(args.top_n_outliers).iterrows():
        print(f"  {row['probe']:<12} {row['section']:<6} {row['polar']:<8} "
              f"{row['FB_ratio']:>10.3f} {row['intensity_back']:>14.0f} {row['intensity_forward']:>14.0f}")

    # ──────────────────────────── GRAFICE ─────────────────────────────────

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(fb, bins=40, color="steelblue", edgecolor="white")
    axes[0].axvline(fb.median(), color="red", linestyle="--", label=f"Median: {fb.median():.2f}")
    axes[0].set_xlabel("F/B ratio (brut)")
    axes[0].set_ylabel("Frecventa")
    axes[0].set_title("Distributie F/B brut")
    axes[0].legend()

    axes[1].hist(log_fb, bins=40, color="seagreen", edgecolor="white")
    axes[1].axvline(log_fb.median(), color="red", linestyle="--", label=f"Median: {log_fb.median():.2f}")
    axes[1].axvline(0, color="gray", linestyle=":", label="F/B = 1 (echilibru)")
    axes[1].set_xlabel("log(F/B)")
    axes[1].set_ylabel("Frecventa")
    axes[1].set_title("Distributie log(F/B) -- simetrizata")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / "fb_ratio_distribution.png", dpi=150)
    print(f"\n  Histograme salvate in {output_dir / 'fb_ratio_distribution.png'}")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["intensity_back"], df["FB_ratio"], alpha=0.5, s=15, color="steelblue")
    ax.set_xlabel("Intensitate totala backward (ch00)")
    ax.set_ylabel("F/B ratio")
    ax.set_title(f"F/B vs Intensitate Backward\n(Pearson r = {corr:.3f})")
    ax.axhline(fb.median(), color="red", linestyle="--", alpha=0.5, label=f"Median F/B: {fb.median():.2f}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "fb_ratio_vs_intensity_back.png", dpi=150)
    print(f"  Scatter plot salvat in {output_dir / 'fb_ratio_vs_intensity_back.png'}")

    plt.close("all")

    # ──────────────────────────── RECOMANDARE ─────────────────────────────

    print(f"\n{'='*60}")
    print(f"  RECOMANDARE")
    print(f"{'='*60}")
    if abs(corr_abs) > 0.5:
        print(f"  Corelatia F/B <-> 1/intensitate_back este puternica ({corr_abs:.2f}).")
        print(f"  CONFIRMA ipoteza profesorului: valorile extreme de F/B sunt")
        print(f"  probabil artefacte ale unei intensitati backward foarte mici,")
        print(f"  nu semnal biologic real.")
        print(f"  Recomandare: foloseste log(F/B) in loc de F/B brut (simetrizeaza")
        print(f"  distributia), SAU exclude perechile cu intensitate_back sub un")
        print(f"  prag minim (similar cu CAF_THRESHOLD din features_caf.py).")
    else:
        print(f"  Corelatia F/B <-> 1/intensitate_back este slaba ({corr_abs:.2f}).")
        print(f"  Valorile extreme de F/B NU par sa fie cauzate predominant de")
        print(f"  intensitate backward mica -- posibil semnal biologic real sau")
        print(f"  alta sursa de variatie. Recomandare: inspecteaza manual imaginile")
        print(f"  din lista de outlieri de mai sus inainte de a decide.")

    print(f"{'='*60}\n")