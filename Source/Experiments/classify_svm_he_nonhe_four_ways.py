"""
classify_svm_he_nonhe_four_ways.py
Replica experimentului HE vs non-HE (Sectiunea 10.1) pentru fiecare din
cele 4 metode de folosire a canalelor SHG testate in Sectiunea 11.10:
Forward, Backward, F+B Combinat, Raport R normalizat.

Pentru fiecare metoda, se evalueaza cele 4 scenarii standard:
    HE -> HE           (antrenat pe tesut colorat, testat pe colorat)
    non-HE -> non-HE    (antrenat pe tesut necolorat, testat pe necolorat)
    HE -> non-HE        (antrenat pe colorat, testat pe necolorat)
    non-HE -> HE        (antrenat pe necolorat, testat pe colorat)

METODOLOGIE (identica cu fix-ul din Sectiunea 10.1.1, extinsa acum la
4 metode simultan): UN SINGUR StratifiedKFold per repetare, aplicat pe
cele 18 grupuri de tesut normale/maligne. Acelasi split de grupuri e
refolosit pentru toate cele 4 metode x 4 scenarii ale unei repetari --
filtrarea pe HE/non-HE (si, dupa caz, pe canal) se aplica DOAR dupa ce
grupurile de train/test sunt deja fixate. Aceasta garanteaza 0% leakage
prin constructie si comparabilitate perfecta: singura sursa de variatie
intre cele 16 rezultate e metoda si scenariul, nu impartirea aleatorie
a datelor.

Utilizare:
    python classify_svm_he_nonhe_four_ways.py --features-csv "...\\features.csv" --r-csv "...\\features_R_ratio.csv"
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
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

DEFAULT_FEATURES_CSV = None
DEFAULT_R_CSV         = None
DEFAULT_PLOTS_DIR     = None
try:
    from config import CSV_FEATURES, CSV_DIR, PLOTS_DIR as CONFIG_PLOTS_DIR, DIAGNOSIS
    DEFAULT_FEATURES_CSV = CSV_FEATURES
    DEFAULT_R_CSV         = CSV_DIR / "features_R_ratio.csv"
    DEFAULT_PLOTS_DIR     = CONFIG_PLOTS_DIR / "svm_he_nonhe_four_ways"
except ImportError:
    DIAGNOSIS = {
        "72066_1" : "normal", "72066_2" : "malign", "72066_3" : "normal",
        "83064_1" : "normal", "83064_2" : "malign", "83064_3" : "malign",
        "89376_1" : "malign", "89376_2" : "malign", "89376_3" : "displazic",
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

N_FOLDS      = 3
KERNEL       = "rbf"
RANDOM_STATE = 42

CHANNEL_BACK    = "ch00"
CHANNEL_FORWARD = "ch02"

METHODS = ["forward", "backward", "combined", "R_ratio"]
SCENARIOS = [
    ("HE", "HE"),
    ("nonHE", "nonHE"),
    ("HE", "nonHE"),
    ("nonHE", "HE"),
]


def build_feature_columns(distances):
    return [
        "Mean", "StDev", "Skew", "Kurt",
    ] + [
        f"{param}_{pair}_d{dist}"
        for pair in DIRECTION_PAIRS
        for dist in distances
        for param in GLCM_PARAMS
    ]


BASE_FEATURE_COLUMNS = build_feature_columns(DISTANCES)


# ──────────────────────────── INCARCARE + PREGATIRE DATE ────────────────────

def add_diagnosis(df):
    df = df.copy()
    df["section_norm"] = df["section"].astype(str).str[0]
    df["probe_base"]   = df["probe"].astype(str).str.replace("HE", "", regex=False)
    df["key"]          = df["probe_base"] + "_" + df["section_norm"]
    df["diagnosis"]    = df["key"].map(DIAGNOSIS)
    df = df.dropna(subset=["diagnosis"])
    df = df[df["diagnosis"] != "displazic"]
    return df


def load_all_datasets(features_csv: Path, r_csv: Path):
    if not features_csv.exists():
        print(f"[EROARE] Nu gasesc {features_csv}")
        sys.exit(1)
    if not r_csv.exists():
        print(f"[EROARE] Nu gasesc {r_csv}. Ruleaza intai features_R_ratio.py")
        sys.exit(1)

    raw_df = pd.read_csv(features_csv)
    raw_df = add_diagnosis(raw_df)

    forward_df  = raw_df[raw_df["channel"] == CHANNEL_FORWARD].copy()
    backward_df = raw_df[raw_df["channel"] == CHANNEL_BACK].copy()

    # Combinat: potrivire pe aceeasi achizitie fizica (probe+section+polar),
    # is_HE identic pentru ambele canale ale aceleiasi achizitii
    back_renamed    = backward_df.rename(columns={c: f"{c}_back" for c in BASE_FEATURE_COLUMNS})
    forward_renamed = forward_df.rename(columns={c: f"{c}_forward" for c in BASE_FEATURE_COLUMNS})
    merge_keys = ["probe", "section", "polar", "key", "diagnosis", "is_HE"]
    combined_df = pd.merge(
        back_renamed[merge_keys + [f"{c}_back" for c in BASE_FEATURE_COLUMNS]],
        forward_renamed[merge_keys + [f"{c}_forward" for c in BASE_FEATURE_COLUMNS]],
        on=merge_keys, how="inner"
    )
    combined_feature_columns = (
        [f"{c}_back" for c in BASE_FEATURE_COLUMNS] +
        [f"{c}_forward" for c in BASE_FEATURE_COLUMNS]
    )

    r_df = pd.read_csv(r_csv)
    r_df = add_diagnosis(r_df)

    datasets = {
        "forward" : (forward_df, BASE_FEATURE_COLUMNS),
        "backward": (backward_df, BASE_FEATURE_COLUMNS),
        "combined": (combined_df, combined_feature_columns),
        "R_ratio" : (r_df, BASE_FEATURE_COLUMNS),
    }

    print(f"\n  Randuri disponibile per metoda:")
    for name, (df, cols) in datasets.items():
        n_he    = (df["is_HE"] == True).sum()
        n_nonhe = (df["is_HE"] == False).sum()
        print(f"    {name:<10}: {len(df):>5} randuri | {df['key'].nunique()} grupuri | "
              f"{len(cols)} parametri | HE: {n_he} | non-HE: {n_nonhe}")

    return datasets


# ──────────────────────────── CROSS-VALIDATION UNIFICATA ────────────────────

def cross_validate_all(datasets, n_repeats):
    """
    UN SINGUR StratifiedKFold per repetare, pe cele 18 grupuri de referinta.
    Toate cele 4 metode x 4 scenarii refolosesc EXACT aceleasi
    train_groups/test_groups la fiecare fold.
    """
    all_keys    = sorted(DIAGNOSIS.keys())
    all_classes = np.array([DIAGNOSIS[k] for k in all_keys])
    mask        = all_classes != "displazic"
    keys        = np.array(all_keys)[mask]
    classes     = all_classes[mask]

    print(f"\n  Grupuri de referinta: {len(keys)}")
    print(f"  Repetari: {n_repeats} | Fold-uri: {N_FOLDS}\n")

    # results[method][scenario_label] = lista de accuracy-uri
    results = {m: {f"{a}->{b}": [] for a, b in SCENARIOS} for m in METHODS}

    for repeat in range(n_repeats):
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE + repeat)

        for train_idx, test_idx in skf.split(keys, classes):
            train_groups = keys[train_idx]
            test_groups  = keys[test_idx]

            for method in METHODS:
                df, feature_columns = datasets[method]

                for train_label, test_label in SCENARIOS:
                    train_is_he = (train_label == "HE")
                    test_is_he  = (test_label == "HE")

                    train_df = df[df["key"].isin(train_groups) & (df["is_HE"] == train_is_he)]
                    test_df  = df[df["key"].isin(test_groups) & (df["is_HE"] == test_is_he)]

                    if train_df["diagnosis"].nunique() < 2 or len(test_df) == 0:
                        continue

                    X_train = train_df[feature_columns].astype(float)
                    y_train = train_df["diagnosis"]
                    X_test  = test_df[feature_columns].astype(float)
                    y_test  = test_df["diagnosis"]

                    scaler  = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test  = scaler.transform(X_test)

                    model = SVC(kernel=KERNEL, class_weight="balanced", random_state=RANDOM_STATE)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    scenario_key = f"{train_label}->{test_label}"
                    results[method][scenario_key].append(accuracy_score(y_test, y_pred))

        if (repeat + 1) % 5 == 0:
            print(f"  Repetarea {repeat+1:2d}/{n_repeats} — "
                  f"R_ratio HE->HE: {np.mean(results['R_ratio']['HE->HE'])*100:.1f}% "
                  f"(monitorizare progres)")

    return results


# ──────────────────────────── VIZUALIZARE ───────────────────────────────────

def plot_grouped_comparison(results, output_dir, n_repeats):
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario_labels = [f"{a}->{b}" for a, b in SCENARIOS]
    method_labels   = {"forward": "Forward", "backward": "Backward",
                        "combined": "F+B Combinat", "R_ratio": "Raport R"}
    colors = ["steelblue", "steelblue", "darkorange", "darkorange"]

    fig, axes = plt.subplots(1, 4, figsize=(20, 6), sharey=True)

    for ax, method in zip(axes, METHODS):
        means = [np.mean(results[method][s]) * 100 if results[method][s] else 0 for s in scenario_labels]
        stds  = [np.std(results[method][s]) * 100 if results[method][s] else 0 for s in scenario_labels]

        bars = ax.bar(scenario_labels, means, yerr=stds, capsize=5, color=colors)
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                     f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.axhline(50, color="red", linestyle="--", linewidth=0.8)
        ax.set_title(method_labels[method], fontsize=12)
        ax.set_xticklabels(scenario_labels, rotation=45, ha="right", fontsize=9)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Accuracy medie (%)")
    fig.suptitle(f"SVM rbf — HE vs non-HE, replicat pentru 4 metode\n"
                 f"{n_repeats}x{N_FOLDS}-Fold CV, acelasi split de grupuri pentru toate combinatiile",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(output_dir / "he_nonhe_four_ways.png", dpi=150)
    print(f"\n  Grafic salvat in {output_dir / 'he_nonhe_four_ways.png'}")


# ──────────────────────────── ENTRY POINT ───────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Comparatie HE vs non-HE replicata pentru 4 metode (Forward/Backward/Combinat/Raport R)"
    )
    parser.add_argument("--features-csv", type=Path, default=DEFAULT_FEATURES_CSV,
                         required=DEFAULT_FEATURES_CSV is None)
    parser.add_argument("--r-csv", type=Path, default=DEFAULT_R_CSV,
                         required=DEFAULT_R_CSV is None)
    parser.add_argument("--n-repeats", type=int, default=25,
                         help="Repetari CV (implicit 25; foloseste o valoare mica "
                              "ex. 3-5 pentru test rapid inainte de rularea completa)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PLOTS_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_dir = args.output_dir or (args.features_csv.parent.parent / "Plots" / "svm_he_nonhe_four_ways")

    print(f"\n{'='*60}")
    print(f"  SVM {KERNEL} — HE vs non-HE, replicat pentru 4 metode")
    print(f"{'='*60}")

    datasets = load_all_datasets(args.features_csv, args.r_csv)

    results = cross_validate_all(datasets, args.n_repeats)

    print(f"\n{'='*60}")
    print(f"  REZULTATE FINALE (16 combinatii: 4 metode x 4 scenarii)")
    print(f"{'='*60}")
    scenario_labels = [f"{a}->{b}" for a, b in SCENARIOS]
    header = f"  {'Metoda':<12}" + "".join(f"{s:>14}" for s in scenario_labels)
    print(header)
    print(f"  {'-'*len(header)}")
    for method in METHODS:
        row = f"  {method:<12}"
        for s in scenario_labels:
            vals = results[method][s]
            row += f"{np.mean(vals)*100:>13.1f}%" if vals else f"{'N/A':>14}"
        print(row)

    plot_grouped_comparison(results, output_dir, args.n_repeats)
    plt.close("all")

    print(f"\n{'='*60}")
    print(f"  Gata! Grafic salvat in: {output_dir}")
    print(f"{'='*60}\n")