"""
plot_correlation_heatmap.py
Genereaza 3 heatmap-uri de corelatie Pearson separate, cate unul
pentru fiecare grup de parametri:
  1. Histograma  (4 parametri:  Mean, StDev, Skew, Kurt)
  2. GLCM        (60 parametri: ASM/Contrast/Correlation/IDM/Entropy
                                x 4 directii x 3 distante)
  3. GLRLM       (20 parametri: SRE/LRE/GLN/RLN/RP x 4 directii)

Coeficientul Pearson variaza intre -1 si +1:
  +1 = corelatie directa perfecta (rosu)
   0 = fara corelatie (alb)
  -1 = corelatie inversa perfecta (albastru)

Perechile mentionate de profesor (ASM<->Entropy invers corelate,
Contrast<->IDM invers corelate) ar trebui sa fie clar vizibile ca
zone albastre in heatmap-ul GLCM.

Utilizare:
    python plot_correlation_heatmap.py
    python plot_correlation_heatmap.py --csv "C:\\...\\features.csv"
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ──────────────────────────── CONFIGURARE ───────────────────────────────────

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
from config import CSV_FEATURES_GLRLM as DEFAULT_CSV, PLOTS_DIR as _PLOTS_BASE
PLOTS_DIR = _PLOTS_BASE / "correlation"

# Coloane de metadate care nu sunt parametri
EXCLUDED_COLUMNS = {
    "probe", "section", "polar", "channel", "is_HE", "filepath",
    "patch_row", "patch_col", "section_norm", "probe_base",
    "key", "diagnosis", "group_id", "dominant_class", "polar_num",
}

DIRECTION_PAIRS = ["0_180", "90_270", "45_225", "135_315"]
GLCM_PARAMS     = ["ASM", "Contrast", "Correlation", "IDM", "Entropy"]
GLRLM_PARAMS    = ["SRE", "LRE", "GLN", "RLN", "RP"]
DISTANCES       = [1, 10, 30]


# ──────────────────────────── INCARCARE DATE ────────────────────────────────

def load_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        print(f"[EROARE] Nu gasesc CSV-ul la: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    # Convertim toate coloanele non-metadate la float
    for col in df.columns:
        if col not in EXCLUDED_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"  CSV incarcat: {csv_path.name} ({len(df)} randuri)")
    return df


# ──────────────────────────── GRUPAREA PARAMETRILOR ─────────────────────────

def build_groups(df: pd.DataFrame) -> dict:
    """
    Imparte parametrii in cele 3 grupuri, folosind doar distanta d1
    si directia 0_180 pentru GLCM si GLRLM — suficient pentru a
    demonstra corelațiile structurale dintre tipurile de parametri,
    fara zgomotul adus de celelalte distante si directii.
    """
    all_cols = set(df.columns) - EXCLUDED_COLUMNS

    # Grup 1: Histograma — toti cei 4 parametri
    histogram_cols = [c for c in ["Mean", "StDev", "Skew", "Kurt"] if c in all_cols]

    # Grup 2: GLCM — doar d1, directia 0_180 (5 parametri)
    glcm_cols = [
        f"{param}_0_180_d1"
        for param in GLCM_PARAMS
        if f"{param}_0_180_d1" in all_cols
    ]

    # Grup 3: GLRLM — doar directia 0_180 (5 parametri, fara distanta)
    glrlm_cols = [
        f"{param}_0_180"
        for param in GLRLM_PARAMS
        if f"{param}_0_180" in all_cols
    ]

    groups = {"Histograma": histogram_cols, "GLCM (d1, 0_180)": glcm_cols}
    if glrlm_cols:
        groups["GLRLM (0_180)"] = glrlm_cols
    else:
        print("  Nota: parametrii GLRLM nu au fost gasiti in CSV.")

    for name, cols in groups.items():
        print(f"  Grup {name}: {len(cols)} parametri — {cols}")

    return groups


# ──────────────────────────── HEATMAP ───────────────────────────────────────

def plot_heatmap(df: pd.DataFrame, columns: list, group_name: str,
                 plots_dir: Path, csv_stem: str):
    """
    Genereaza un heatmap de corelatie Pearson pentru un grup de parametri.
    Cu matrici mici (4x4, 5x5), afiseaza valorile numerice in fiecare
    celula si foloseste etichete mari, lizibile.
    """
    if not columns:
        print(f"  [Atentie] Niciun parametru gasit pentru grupul {group_name}, skip.")
        return

    corr = df[columns].corr(method="pearson")
    n    = len(columns)

    # Dimensiune fixa, confortabila pentru matrice mici
    fig_size = max(5, n * 0.9)
    fig, ax  = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Coeficient Pearson", fontsize=11)
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])

    # Etichete scurtate — pentru 5 parametri avem loc pentru etichete clare
    def short_label(col):
        parts = col.split("_")
        if len(parts) == 4:
            # GLCM: param_dir1_dir2_dX -> param\n(dir1_dir2, dX)
            return f"{parts[0]}\n({parts[1]}_{parts[2]}, {parts[3]})"
        elif len(parts) == 3:
            # GLRLM: param_dir1_dir2 -> param\n(dir1_dir2)
            return f"{parts[0]}\n({parts[1]}_{parts[2]})"
        return col

    labels = [short_label(c) for c in columns]

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=12)
    ax.set_yticklabels(labels, fontsize=12)

    # Valorile numerice in fiecare celula — intotdeauna, pentru matrice mici
    for i in range(n):
        for j in range(n):
            val   = corr.values[i, j]
            color = "white" if abs(val) > 0.65 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=11, color=color, fontweight="bold")

    ax.set_title(
        f"Corelatie Pearson — {group_name}\n"
        f"Sursa: {csv_stem}",
        fontsize=13, pad=15
    )

    plt.tight_layout()

    plots_dir.mkdir(parents=True, exist_ok=True)
    safe_name = group_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
    out_path  = plots_dir / f"correlation_{safe_name}_{csv_stem}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Heatmap {group_name} salvat in {out_path}")


# ──────────────────────────── ENTRY POINT ───────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Heatmap-uri de corelatie Pearson pe grupuri de parametri SHG"
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV,
                        help=f"Calea catre CSV (implicit: {DEFAULT_CSV})")
    parser.add_argument("--output-dir", type=Path, default=PLOTS_DIR,
                        help=f"Folder pentru grafice (implicit: {PLOTS_DIR})")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"  Heatmap-uri de corelatie Pearson — grupuri de parametri")
    print(f"  CSV: {args.csv}")
    print(f"  Output: {args.output_dir}")
    print(f"{'='*60}\n")

    df     = load_data(args.csv)
    groups = build_groups(df)

    print(f"\n  Generare heatmap-uri...")
    for group_name, columns in groups.items():
        plot_heatmap(df, columns, group_name, args.output_dir, args.csv.stem)

    print(f"\n{'='*60}")
    print(f"  Gata! Graficele sunt in: {args.output_dir}")
    print(f"{'='*60}\n")