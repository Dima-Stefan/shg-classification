"""
classify_svm_he_nonhe_four_ways_3class.py
Extensie la 3 CLASE (normal/malign/displazic) a comparatiei HE vs
non-HE replicate pentru cele 4 metode de folosire a canalelor SHG
(Forward, Backward, F+B Combinat, Raport R) -- vezi
classify_svm_he_nonhe_four_ways.py pentru varianta binara.

ATENTIE -- limitare de date: doar 5 grupuri de tesut displazic exista
in total, impartite atat intre cele 3 clase (pentru stratificare CV)
cat si intre HE/non-HE (pentru cele 4 scenarii). Unele combinatii
metoda x scenariu vor avea foarte putine exemple displazice de test la
anumite fold-uri -- rezultatele pe displazic vor avea variatie mare si
trebuie interpretate cu prudenta. Fold-urile unde train-ul nu contine
toate cele 3 clase sunt sarite automat (raportat in consola).

METODOLOGIE: identica cu varianta binara -- UN SINGUR StratifiedKFold
per repetare, aplicat pe toate cele 23 de grupuri de tesut (stratificat
pe cele 3 clase), refolosit pentru toate cele 4 metode x 4 scenarii.
Filtrarea HE/non-HE se aplica DOAR dupa fixarea grupurilor de
train/test. class_weight='balanced' pentru compensarea dezechilibrului
9 normal / 9 malign / 5 displazic.

Output:
    - Tabel text cu balanced accuracy pentru toate cele 16 combinatii
    - 16 confusion matrix (una per metoda x scenariu), salvate separat

Utilizare:
    python classify_svm_he_nonhe_four_ways_3class.py --features-csv "...\\features.csv" --r-csv "...\\features_R_ratio.csv"
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
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    confusion_matrix, ConfusionMatrixDisplay,
)

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
    DEFAULT_PLOTS_DIR     = CONFIG_PLOTS_DIR / "svm_he_nonhe_four_ways_3class"
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

CLASSES   = ["normal", "malign", "displazic"]
METHODS   = ["forward", "backward", "combined", "R_ratio"]
SCENARIOS = [
    ("HE", "HE"),
    ("nonHE", "nonHE"),
    ("HE", "nonHE"),
    ("nonHE", "HE"),
]
METHOD_LABELS = {"forward": "Forward", "backward": "Backward",
                  "combined": "F+B_Combinat", "R_ratio": "Raport_R"}


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
    """Identic cu varianta binara, dar PASTREAZA displazicul."""
    df = df.copy()
    df["section_norm"] = df["section"].astype(str).str[0]
    df["probe_base"]   = df["probe"].astype(str).str.replace("HE", "", regex=False)
    df["key"]          = df["probe_base"] + "_" + df["section_norm"]
    df["diagnosis"]    = df["key"].map(DIAGNOSIS)
    df = df.dropna(subset=["diagnosis"])
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

    print(f"\n  Randuri disponibile per metoda (3-clase):")
    for name, (df, cols) in datasets.items():
        n_he    = (df["is_HE"] == True).sum()
        n_nonhe = (df["is_HE"] == False).sum()
        n_disp  = (df["diagnosis"] == "displazic").sum()
        print(f"    {name:<10}: {len(df):>5} randuri | {df['key'].nunique()} grupuri | "
              f"{len(cols)} parametri | HE: {n_he} | non-HE: {n_nonhe} | displazic: {n_disp}")

    return datasets


# ──────────────────────────── CROSS-VALIDATION UNIFICATA ────────────────────

def cross_validate_all(datasets, n_repeats):
    all_keys    = sorted(DIAGNOSIS.keys())
    classes_all = np.array([DIAGNOSIS[k] for k in all_keys])
    keys        = np.array(all_keys)

    print(f"\n  Grupuri de referinta (toate 3 clasele): {len(keys)}")
    for c in CLASSES:
        print(f"    {c}: {(classes_all == c).sum()}")
    print(f"  Repetari: {n_repeats} | Fold-uri: {N_FOLDS}\n")

    bal_acc_results = {m: {f"{a}->{b}": [] for a, b in SCENARIOS} for m in METHODS}
    acc_results     = {m: {f"{a}->{b}": [] for a, b in SCENARIOS} for m in METHODS}
    y_true_agg = {m: {f"{a}->{b}": [] for a, b in SCENARIOS} for m in METHODS}
    y_pred_agg = {m: {f"{a}->{b}": [] for a, b in SCENARIOS} for m in METHODS}
    n_skipped  = {m: {f"{a}->{b}": 0 for a, b in SCENARIOS} for m in METHODS}

    for repeat in range(n_repeats):
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE + repeat)

        for train_idx, test_idx in skf.split(keys, classes_all):
            train_groups = keys[train_idx]
            test_groups  = keys[test_idx]

            for method in METHODS:
                df, feature_columns = datasets[method]

                for train_label, test_label in SCENARIOS:
                    train_is_he = (train_label == "HE")
                    test_is_he  = (test_label == "HE")
                    scenario_key = f"{train_label}->{test_label}"

                    train_df = df[df["key"].isin(train_groups) & (df["is_HE"] == train_is_he)]
                    test_df  = df[df["key"].isin(test_groups) & (df["is_HE"] == test_is_he)]

                    if train_df["diagnosis"].nunique() < 3 or len(test_df) == 0:
                        n_skipped[method][scenario_key] += 1
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

                    acc_results[method][scenario_key].append(accuracy_score(y_test, y_pred))
                    bal_acc_results[method][scenario_key].append(
                        balanced_accuracy_score(y_test, y_pred)
                    )
                    y_true_agg[method][scenario_key].extend(y_test.tolist())
                    y_pred_agg[method][scenario_key].extend(y_pred.tolist())

        if (repeat + 1) % 5 == 0:
            print(f"  Repetarea {repeat+1:2d}/{n_repeats} — "
                  f"R_ratio HE->HE bal.acc: "
                  f"{np.mean(bal_acc_results['R_ratio']['HE->HE'])*100:.1f}% (monitorizare)")

    return acc_results, bal_acc_results, y_true_agg, y_pred_agg, n_skipped


# ──────────────────────────── CONFUSION MATRICES (16) ────────────────────────

def plot_all_confusion_matrices(y_true_agg, y_pred_agg, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for method in METHODS:
        for train_label, test_label in SCENARIOS:
            scenario_key = f"{train_label}->{test_label}"
            y_true = y_true_agg[method][scenario_key]
            y_pred = y_pred_agg[method][scenario_key]

            if len(y_true) == 0:
                print(f"  [SARIT] {method} {scenario_key}: nicio evaluare valida")
                continue

            cm   = confusion_matrix(y_true, y_pred, labels=CLASSES)
            disp = ConfusionMatrixDisplay(cm, display_labels=[c.capitalize() for c in CLASSES])
            fig, ax = plt.subplots(figsize=(6, 5))
            disp.plot(ax=ax, colorbar=False, cmap="Blues")
            ax.set_title(f"{METHOD_LABELS[method]} — {scenario_key}", fontsize=12)
            plt.tight_layout()

            filename = f"cm_{method}_{train_label}-{test_label}.png"
            plt.savefig(output_dir / filename, dpi=150)
            plt.close(fig)

    print(f"\n  Toate confusion matrix salvate in: {output_dir}")


# ──────────────────────────── VIZUALIZARE SUMAR ──────────────────────────────

def plot_grouped_comparison(bal_acc_results, output_dir, n_repeats):
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario_labels = [f"{a}->{b}" for a, b in SCENARIOS]
    colors = ["steelblue", "steelblue", "darkorange", "darkorange"]

    fig, axes = plt.subplots(1, 4, figsize=(20, 6), sharey=True)

    for ax, method in zip(axes, METHODS):
        means = [np.mean(bal_acc_results[method][s]) * 100 if bal_acc_results[method][s] else 0
                 for s in scenario_labels]
        stds  = [np.std(bal_acc_results[method][s]) * 100 if bal_acc_results[method][s] else 0
                 for s in scenario_labels]

        bars = ax.bar(scenario_labels, means, yerr=stds, capsize=5, color=colors)
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                     f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.axhline(100/3, color="red", linestyle="--", linewidth=0.8, label="Random (33.3%)")
        ax.set_title(METHOD_LABELS[method], fontsize=12)
        ax.set_xticklabels(scenario_labels, rotation=45, ha="right", fontsize=9)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Balanced Accuracy medie (%)")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"SVM rbf — HE vs non-HE, 3-clase, replicat pentru 4 metode\n"
                 f"{n_repeats}x{N_FOLDS}-Fold CV, acelasi split de grupuri pentru toate combinatiile",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(output_dir / "he_nonhe_four_ways_3class_summary.png", dpi=150)
    print(f"\n  Grafic sumar salvat in {output_dir / 'he_nonhe_four_ways_3class_summary.png'}")


# ──────────────────────────── ENTRY POINT ───────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Comparatie HE vs non-HE 3-clase, replicata pentru 4 metode, cu confusion matrix complete"
    )
    parser.add_argument("--features-csv", type=Path, default=DEFAULT_FEATURES_CSV,
                         required=DEFAULT_FEATURES_CSV is None)
    parser.add_argument("--r-csv", type=Path, default=DEFAULT_R_CSV,
                         required=DEFAULT_R_CSV is None)
    parser.add_argument("--n-repeats", type=int, default=25,
                         help="Repetari CV (implicit 25; foloseste o valoare mica pentru test rapid)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PLOTS_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_dir = args.output_dir or (args.features_csv.parent.parent / "Plots" / "svm_he_nonhe_four_ways_3class")

    print(f"\n{'='*60}")
    print(f"  SVM {KERNEL} — HE vs non-HE, 3-CLASE, replicat pentru 4 metode")
    print(f"{'='*60}")

    datasets = load_all_datasets(args.features_csv, args.r_csv)

    acc_results, bal_acc_results, y_true_agg, y_pred_agg, n_skipped = cross_validate_all(
        datasets, args.n_repeats
    )

    print(f"\n{'='*60}")
    print(f"  REZULTATE FINALE — Balanced Accuracy (16 combinatii)")
    print(f"{'='*60}")
    scenario_labels = [f"{a}->{b}" for a, b in SCENARIOS]
    header = f"  {'Metoda':<12}" + "".join(f"{s:>14}" for s in scenario_labels)
    print(header)
    print(f"  {'-'*len(header)}")
    for method in METHODS:
        row = f"  {method:<12}"
        for s in scenario_labels:
            vals = bal_acc_results[method][s]
            row += f"{np.mean(vals)*100:>13.1f}%" if vals else f"{'N/A':>14}"
        print(row)

    print(f"\n  Fold-uri sarite (train fara toate cele 3 clase, sau test gol):")
    for method in METHODS:
        for s in scenario_labels:
            if n_skipped[method][s] > 0:
                print(f"    {method:<10} {s:<12}: {n_skipped[method][s]} sarite")

    print(f"\n  Generare 16 confusion matrix...")
    plot_all_confusion_matrices(y_true_agg, y_pred_agg, output_dir)

    plot_grouped_comparison(bal_acc_results, output_dir, args.n_repeats)
    plt.close("all")

    print(f"\n{'='*60}")
    print(f"  Gata! Toate graficele sunt in: {output_dir}")
    print(f"{'='*60}\n")