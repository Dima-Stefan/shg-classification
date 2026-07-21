"""
test_rf_hyperparams.py
Testeaza o grila de combinatii de hiperparametri Random Forest
(n_estimators x max_depth x min_samples_leaf), pentru a identifica
configuratia optima si a verifica daca performanta mai slaba fata de
SVM (63.7% vs 69.4%, Sectiune RF baseline) se datoreaza overfitting-ului
arborilor nelimitati (max_depth=None) pe setul de date mic (18 grupuri
de tesut la clasificarea binara).

Foloseste EXACT aceeasi metodologie SVM->RF adaptata: distante GLCM
d1+d10+d30 (64 parametri), SVM rbf inlocuit cu RandomForestClassifier,
25x3-Fold CV la nivel de grup de tesut, identic cu classify_rf_kfold.py
si test_distance_combinations.py.

Grila testata (implicit):
    n_estimators    : [100, 300, 500]
    max_depth       : [None, 5, 10, 20]
    min_samples_leaf: [1, 2, 5]
    Total: 3 x 4 x 3 = 36 combinatii

Output:
    - Tabel complet cu toate combinatiile, sortat descrescator dupa accuracy
    - Grafic bar chart cu top 10 combinatii
    - CSV cu rezultatele complete (rf_hyperparams_results.csv)

Utilizare:
    python test_rf_hyperparams.py
    python test_rf_hyperparams.py --n-repeats 5   (rulare rapida de test)
"""

import argparse
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

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

try:
    from config import CSV_FEATURES, PLOTS_DIR, DIAGNOSIS
    INPUT_CSV  = CSV_FEATURES
    OUTPUT_DIR = PLOTS_DIR / "rf_hyperparams"
except ImportError:
    INPUT_CSV  = Path(r"C:\Users\dimas\Downloads\projects\venv\Results\CSV\features.csv")
    OUTPUT_DIR = Path(r"C:\Users\dimas\Downloads\projects\venv\Results\Plots\rf_hyperparams")
    DIAGNOSIS = {
        "72066_1": "normal", "72066_2": "malign", "72066_3": "normal",
        "83064_1": "normal", "83064_2": "malign", "83064_3": "malign",
        "89376_1": "malign", "89376_2": "malign", "89376_3": "displazic",
        "102312_1": "normal", "102312_2": "malign", "102312_3": "malign",
        "149113_1": "displazic", "149113_2": "displazic", "149113_3": "displazic",
        "155820_1": "malign", "155820_2": "malign", "155820_3": "displazic",
        "156222_1": "normal", "156222_2": "normal", "156222_3": "normal",
        "182467_1": "normal", "182467_3": "normal",
    }

# ──────────────────────────── CONFIGURARE ───────────────────────────────────

DIRECTION_PAIRS = ["0_180", "90_270", "45_225", "135_315"]
GLCM_PARAMS     = ["ASM", "Contrast", "Correlation", "IDM", "Entropy"]
DISTANCES       = [1, 10, 30]

RANDOM_STATE = 42
N_FOLDS      = 3

# Grila de hiperparametri testata
N_ESTIMATORS_GRID     = [100, 300, 500]
MAX_DEPTH_GRID        = [None, 5, 10, 20]
MIN_SAMPLES_LEAF_GRID = [1, 2, 5]


def build_feature_columns(distances):
    return [
        "Mean", "StDev", "Skew", "Kurt",
    ] + [
        f"{param}_{pair}_d{dist}"
        for pair in DIRECTION_PAIRS
        for dist in distances
        for param in GLCM_PARAMS
    ]


FEATURE_COLUMNS = build_feature_columns(DISTANCES)


# ──────────────────────────── INCARCARE DATE ────────────────────────────────

def load_data(csv_path: Path):
    if not csv_path.exists():
        print(f"[EROARE] Nu gasesc CSV-ul la: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    df["section_norm"] = df["section"].astype(str).str[0]
    df["probe_base"]   = df["probe"].astype(str).str.replace("HE", "", regex=False)
    df["key"]          = df["probe_base"] + "_" + df["section_norm"]
    df["diagnosis"]    = df["key"].map(DIAGNOSIS)

    df = df.dropna(subset=["diagnosis"])
    df = df[df["diagnosis"] != "displazic"]

    print(f"  CSV incarcat: {csv_path.name}")
    print(f"  Total imagini: {len(df)} | "
          f"normal: {(df['diagnosis']=='normal').sum()} | "
          f"malign: {(df['diagnosis']=='malign').sum()}")

    return df


# ──────────────────────────── CROSS-VALIDATION ──────────────────────────────

def cross_validate(df, n_estimators, max_depth, min_samples_leaf, n_repeats):
    probe_sections = (
        df.groupby("key")["diagnosis"]
        .agg(lambda x: x.value_counts().index[0])
        .reset_index()
        .rename(columns={"diagnosis": "dominant_class"})
    )
    keys    = probe_sections["key"].values
    classes = probe_sections["dominant_class"].values

    results = []

    for repeat in range(n_repeats):
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                               random_state=RANDOM_STATE + repeat)

        for train_idx, test_idx in skf.split(keys, classes):
            train_keys = keys[train_idx]
            test_keys  = keys[test_idx]

            train_df = df[df["key"].isin(train_keys)]
            test_df  = df[df["key"].isin(test_keys)]

            if train_df["diagnosis"].nunique() < 2 or len(test_df) == 0:
                continue

            X_train = train_df[FEATURE_COLUMNS].astype(float)
            y_train = train_df["diagnosis"]
            X_test  = test_df[FEATURE_COLUMNS].astype(float)
            y_test  = test_df["diagnosis"]

            scaler  = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test  = scaler.transform(X_test)

            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            results.append(accuracy_score(y_test, y_pred))

    return results


# ──────────────────────────── TESTARE GRILA COMPLETA ────────────────────────

def test_all_combinations(df, n_repeats):
    combos = list(product(N_ESTIMATORS_GRID, MAX_DEPTH_GRID, MIN_SAMPLES_LEAF_GRID))
    print(f"\n  Testare {len(combos)} combinatii de hiperparametri")
    print(f"  n_estimators     : {N_ESTIMATORS_GRID}")
    print(f"  max_depth        : {MAX_DEPTH_GRID}")
    print(f"  min_samples_leaf : {MIN_SAMPLES_LEAF_GRID}")
    print(f"  {n_repeats}x{N_FOLDS}-Fold CV per combinatie "
          f"({n_repeats * N_FOLDS} evaluari x {len(combos)} combinatii = "
          f"{n_repeats * N_FOLDS * len(combos)} antrenari RF totale)\n")

    rows = []
    for n_est, max_d, min_leaf in tqdm(combos, desc="Combinatii hiperparametri", unit="combo"):
        results = cross_validate(df, n_est, max_d, min_leaf, n_repeats)

        max_d_label = "None" if max_d is None else str(max_d)
        rows.append({
            "config"           : f"n{n_est}_d{max_d_label}_leaf{min_leaf}",
            "n_estimators"     : n_est,
            "max_depth"        : max_d_label,
            "min_samples_leaf" : min_leaf,
            "accuracy_medie"   : np.mean(results) * 100,
            "accuracy_std"     : np.std(results) * 100,
            "accuracy_min"     : np.min(results) * 100,
            "accuracy_max"     : np.max(results) * 100,
        })

    results_df = pd.DataFrame(rows).sort_values(
        "accuracy_medie", ascending=False
    ).reset_index(drop=True)

    return results_df


# ──────────────────────────── VIZUALIZARE ───────────────────────────────────

def plot_top_combinations(results_df, output_dir, top_n=10):
    output_dir.mkdir(parents=True, exist_ok=True)
    top = results_df.head(top_n)

    fig, ax = plt.subplots(figsize=(13, 6))
    bars = ax.bar(top["config"], top["accuracy_medie"],
                   yerr=top["accuracy_std"], color="mediumseagreen", capsize=4)

    for bar, val in zip(bars, top["accuracy_medie"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.axhline(63.7, color="gray", linestyle=":", linewidth=1.2,
               label="RF baseline (n_estimators=500, max_depth=None): 63.7%")
    ax.axhline(69.4, color="red", linestyle="--", linewidth=1.2,
               label="SVM rbf (referinta): 69.4%")
    ax.set_ylabel("Accuracy medie (%)")
    ax.set_title(f"Top {top_n} combinatii de hiperparametri Random Forest")
    ax.set_xticklabels(top["config"], rotation=45, ha="right")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out_path = output_dir / "top_rf_hyperparams.png"
    plt.savefig(out_path, dpi=150)
    print(f"\n  Grafic salvat in {out_path}")


# ──────────────────────────── ENTRY POINT ───────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Testeaza o grila de hiperparametri Random Forest (n_estimators x max_depth x min_samples_leaf)"
    )
    parser.add_argument("--csv", type=Path, default=INPUT_CSV)
    parser.add_argument("--n-repeats", type=int, default=25,
                         help="Repetari CV per combinatie (implicit 25; "
                              "foloseste o valoare mica ex. 5 pentru test rapid)")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"  Testare exhaustiva hiperparametri Random Forest")
    print(f"{'='*60}")

    df = load_data(args.csv)
    results_df = test_all_combinations(df, args.n_repeats)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_csv_path = args.output_dir / "rf_hyperparams_results.csv"
    results_df.to_csv(results_csv_path, index=False)

    print(f"\n{'='*60}")
    print(f"  REZULTATE — Top 10 combinatii")
    print(f"{'='*60}")
    print(results_df.head(10).to_string(index=False))

    baseline_row = results_df[
        (results_df["n_estimators"] == 500) &
        (results_df["max_depth"] == "None") &
        (results_df["min_samples_leaf"] == 1)
    ]
    if not baseline_row.empty:
        rank = results_df.index[
            (results_df["n_estimators"] == 500) &
            (results_df["max_depth"] == "None") &
            (results_df["min_samples_leaf"] == 1)
        ][0] + 1
        print(f"\n  Configuratia baseline (n_estimators=500, max_depth=None, "
              f"min_samples_leaf=1) se claseaza pe locul {rank}/{len(results_df)}")
        print(f"  Accuracy: {baseline_row['accuracy_medie'].values[0]:.1f}%")

    best_acc = results_df.iloc[0]["accuracy_medie"]
    print(f"\n  Cea mai buna configuratie: {results_df.iloc[0]['config']}")
    print(f"  Accuracy: {best_acc:.1f}%")
    print(f"  Diferenta fata de SVM rbf (69.4%): {best_acc - 69.4:+.1f} puncte procentuale")

    plot_top_combinations(results_df, args.output_dir)

    print(f"\n{'='*60}")
    print(f"  Rezultate complete salvate in: {results_csv_path}")
    print(f"{'='*60}\n")