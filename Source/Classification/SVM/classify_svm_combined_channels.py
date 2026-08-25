"""
classify_svm_combined_channels.py
Clasificare SVM folosind parametrii AMBELOR canale SHG (backward ch00 +
forward ch02) SIMULTAN, ca un singur vector de 128 de parametri (64
backward + 64 forward), pentru fiecare achizitie fizica in parte.

Motivatie: comparatia anterioara (classify_svm_channel_comparison.py)
a aratat ca backward si forward, folosite SEPARAT, clasifica bine
fiecare (68.0% / 66.8%), dar un model antrenat pe un canal generalizeaza
slab pe celalalt (54.3% / 59.2%) -- semn ca cele doua canale capteaza
informatie structurala complementara, nu redundanta. Aceasta varianta
testeaza daca folosirea simultana a ambelor canale ca parametri aduce
un castig real fata de fiecare canal separat sau fata de referinta
combinata pe randuri (69.4%, unde fiecare imagine -- indiferent de
canal -- e un exemplu independent).

Potrivire: doua imagini (ch00 si ch02) sunt combinate intr-un singur
rand DOAR daca provin din aceeasi achizitie fizica -- acelasi probe,
aceeasi sectiune, aceeasi polarizare (conventia de numire fisier:
[probe]_[sectiune]-[polarizare]_[canal].tif). Randurile fara pereche
completa (lipseste ch00 sau ch02 pentru acea combinatie) sunt eliminate.

Grupare CV identica cu restul proiectului: la nivel de grup de tesut
(probe+sectiune), pentru a evita orice leakage.

Utilizare:
    python classify_svm_combined_channels.py --csv "...\\features.csv"
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
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
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

DEFAULT_CSV_PATH  = None
DEFAULT_PLOTS_DIR = None
try:
    from config import CSV_FEATURES, PLOTS_DIR as CONFIG_PLOTS_DIR, DIAGNOSIS
    DEFAULT_CSV_PATH  = CSV_FEATURES
    DEFAULT_PLOTS_DIR = CONFIG_PLOTS_DIR / "svm_combined_channels"
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
N_REPEATS    = 25
KERNEL       = "rbf"
RANDOM_STATE = 42

CHANNEL_BACK    = "ch00"
CHANNEL_FORWARD = "ch02"


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


# ──────────────────────────── INCARCARE + COMBINARE CANALE ──────────────────

def load_and_combine(csv_path: Path):
    print(f"\n{'='*60}")
    print(f"  SVM {KERNEL} — Parametri COMBINATI (backward + forward)")
    print(f"  {N_REPEATS}x{N_FOLDS}-Fold CV | Distante GLCM: {DISTANCES}")
    print(f"{'='*60}\n")

    if not csv_path.exists():
        print(f"[EROARE] Nu gasesc CSV-ul la: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"  Total imagini incarcate : {len(df)}")

    df["section_norm"] = df["section"].astype(str).str[0]
    df["probe_base"]   = df["probe"].str.replace("HE", "", regex=False)
    df["key"]          = df["probe_base"] + "_" + df["section_norm"]
    df["diagnosis"]    = df["key"].map(DIAGNOSIS)

    df = df.dropna(subset=["diagnosis"])
    df = df[df["diagnosis"] != "displazic"]

    back_df    = df[df["channel"] == CHANNEL_BACK].copy()
    forward_df = df[df["channel"] == CHANNEL_FORWARD].copy()

    # Redenumim coloanele de parametri cu sufix _back / _forward inainte de merge
    back_df    = back_df.rename(columns={c: f"{c}_back" for c in BASE_FEATURE_COLUMNS})
    forward_df = forward_df.rename(columns={c: f"{c}_forward" for c in BASE_FEATURE_COLUMNS})

    # Potrivire pe achizitia fizica: acelasi probe + sectiune + polarizare
    merge_keys = ["probe", "section", "polar", "key", "diagnosis"]
    back_subset    = back_df[merge_keys + [f"{c}_back" for c in BASE_FEATURE_COLUMNS]]
    forward_subset = forward_df[merge_keys + [f"{c}_forward" for c in BASE_FEATURE_COLUMNS]]

    combined = pd.merge(back_subset, forward_subset, on=merge_keys, how="inner")

    feature_columns = (
        [f"{c}_back" for c in BASE_FEATURE_COLUMNS] +
        [f"{c}_forward" for c in BASE_FEATURE_COLUMNS]
    )

    print(f"  Imagini backward (ch00)  : {len(back_df)}")
    print(f"  Imagini forward (ch02)   : {len(forward_df)}")
    print(f"  Perechi combinate (128 parametri): {len(combined)}")
    print(f"  Normal    : {(combined['diagnosis']=='normal').sum()}")
    print(f"  Malign    : {(combined['diagnosis']=='malign').sum()}")

    return combined, feature_columns


# ──────────────────────────── CROSS-VALIDATION ──────────────────────────────

def cross_validate(df, feature_columns):
    probe_sections = (
        df.groupby("key")["diagnosis"]
        .agg(lambda x: x.value_counts().index[0])
        .reset_index()
        .rename(columns={"diagnosis": "dominant_class"})
    )
    keys    = probe_sections["key"].values
    classes = probe_sections["dominant_class"].values

    print(f"\n  Sectiuni unice : {len(keys)}")
    print(f"  Normal         : {(classes=='normal').sum()}")
    print(f"  Malign         : {(classes=='malign').sum()}")
    print(f"  Repetari       : {N_REPEATS}")
    print(f"  Fold-uri       : {N_FOLDS}\n")

    results    = []
    all_y_true = []
    all_y_pred = []

    for repeat in range(N_REPEATS):
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                               random_state=RANDOM_STATE + repeat)

        for train_idx, test_idx in skf.split(keys, classes):
            train_keys = keys[train_idx]
            test_keys  = keys[test_idx]

            train_df = df[df["key"].isin(train_keys)]
            test_df  = df[df["key"].isin(test_keys)]

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

            results.append(accuracy_score(y_test, y_pred))
            all_y_true.extend(y_test.tolist())
            all_y_pred.extend(y_pred.tolist())

        if (repeat + 1) % 5 == 0:
            print(f"  Repetarea {repeat+1:2d}/{N_REPEATS} — accuracy medie: {np.mean(results)*100:.1f}%")

    return results, all_y_true, all_y_pred


# ──────────────────────────── VIZUALIZARI ───────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    cm   = confusion_matrix(y_true, y_pred, labels=["normal", "malign"])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Malign"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — SVM {KERNEL} (128 param., backward+forward)", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix_combined.png", dpi=150)
    print(f"  Confusion matrix salvata")


def plot_feature_importance(df, feature_columns, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    X = df[feature_columns].astype(float)
    y = df["diagnosis"]

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = SVC(kernel=KERNEL, class_weight="balanced", random_state=RANDOM_STATE)
    model.fit(X_scaled, y)

    print(f"\n  Calculare Permutation Importance...")
    perm = permutation_importance(
        model, X_scaled, y, n_repeats=10, random_state=RANDOM_STATE, scoring="accuracy"
    )
    indices = np.argsort(perm.importances_mean)[::-1]
    top_n   = min(20, len(feature_columns))
    top_idx = indices[:top_n]

    colors = ["steelblue" if feature_columns[i].endswith("_back") else "darkorange" for i in top_idx]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(top_n), perm.importances_mean[top_idx],
           yerr=perm.importances_std[top_idx], color=colors, capsize=3)
    ax.set_xticks(range(top_n))
    ax.set_xticklabels([feature_columns[i] for i in top_idx], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Scadere medie accuracy")
    ax.set_title("Permutation Importance — 128 parametri (albastru=backward, portocaliu=forward)",
                 fontsize=12)
    ax.axhline(0, color="red", linestyle="--", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance_combined.png", dpi=150)
    print(f"  Feature importance salvata")

    n_back_top    = sum(1 for i in top_idx if feature_columns[i].endswith("_back"))
    n_forward_top = top_n - n_back_top
    print(f"\n  In top {top_n}: {n_back_top} parametri backward, {n_forward_top} parametri forward")

    print(f"\n  Top 10 parametri:")
    print(f"  {'Parametru':<40} {'Importanta':>12}")
    print(f"  {'-'*54}")
    for i in indices[:10]:
        print(f"  {feature_columns[i]:<40} {perm.importances_mean[i]:>12.4f}")


# ──────────────────────────── ENTRY POINT ───────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Clasificare SVM cu parametri combinati din ambele canale SHG (backward+forward)"
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH,
                         required=DEFAULT_CSV_PATH is None,
                         help="Calea catre CSV-ul de analizat (ex. features.csv)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PLOTS_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_dir = args.output_dir or (args.csv.parent.parent / "Plots" / "svm_combined_channels")

    df, feature_columns = load_and_combine(args.csv)

    results, all_y_true, all_y_pred = cross_validate(df, feature_columns)

    mean_acc = np.mean(results)
    std_acc  = np.std(results)

    print(f"\n{'='*60}")
    print(f"  REZULTATE FINALE — 128 parametri (backward + forward combinati)")
    print(f"{'='*60}")
    print(f"  Accuracy medie  : {mean_acc*100:.1f}% (+/- {std_acc*100:.1f}%)")
    print(f"  Accuracy minima : {min(results)*100:.1f}%")
    print(f"  Accuracy maxima : {max(results)*100:.1f}%")
    print(f"\n  Referinte anterioare:")
    print(f"    Baseline (64 param., toate imaginile amestecate) : 69.4%")
    print(f"    Doar backward (ch00, in-domeniu)                 : 68.0%")
    print(f"    Doar forward (ch02, in-domeniu)                  : 66.8%")

    print(f"\n  Raport detaliat:")
    print(classification_report(all_y_true, all_y_pred, labels=["normal", "malign"], zero_division=0))

    print(f"\n  Generare grafice...")
    plot_confusion_matrix(all_y_true, all_y_pred, output_dir)
    plot_feature_importance(df, feature_columns, output_dir)

    plt.close("all")

    print(f"\n{'='*60}")
    print(f"  Gata! Graficele sunt in: {output_dir}")
    print(f"{'='*60}\n")