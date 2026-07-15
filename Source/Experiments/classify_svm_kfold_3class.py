"""
classify_svm_kfold_3class.py
Clasificarea imaginilor SHG (512x512) in 3 clase: normal, malign,
displazic -- extensie a classify_svm_kfold.py (binar normal/malign).

Foloseste EXACT aceeasi metodologie validata pentru cazul binar:
SVM rbf, class_weight='balanced', StratifiedKFold 3-Fold repetat 25 ori
la nivel de grup de tesut (probe+sectiune), configuratia standard de
distante GLCM d1+d10+d30 (69.4% in cazul binar).

ATENTIE -- dezechilibru sever de clase:
    Normal    : 9 grupuri de tesut
    Malign    : 9 grupuri de tesut
    Displazic : 5 grupuri de tesut

Cu doar 5 grupuri displazice si 3-Fold CV, fiecare fold de test are in
medie 1-2 grupuri displazice -- variatia pe aceasta clasa va fi mare.
class_weight='balanced' compenseaza partial dezechilibrul in antrenare,
dar NU compenseaza lipsa de date -- rezultatele pentru displazic trebuie
interpretate cu prudenta si, daca e posibil, validate ulterior prin
adaugarea de probe noi (Sectiunea 8.1 din documentatie).

Utilizare:
    python classify_svm_kfold_3class.py
"""

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
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
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

try:
    from config import CSV_FEATURES, PLOTS_DIR, DIAGNOSIS
    CSV_PATH  = CSV_FEATURES
    PLOTS_DIR = PLOTS_DIR / "svm_kfold_3class"
except ImportError:
    CSV_PATH  = Path(r"C:\Users\dimas\Downloads\projects\venv\Results\CSV\features.csv")
    PLOTS_DIR = Path(r"C:\Users\dimas\Downloads\projects\venv\Results\Plots\svm_kfold_3class")
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

N_FOLDS      = 3
N_REPEATS    = 25
KERNEL       = "rbf"
RANDOM_STATE = 42

CLASSES = ["normal", "malign", "displazic"]  # ordine fixa pentru rapoarte/grafice


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

def load_data():
    print(f"\n{'='*60}")
    print(f"  SVM {KERNEL} 3-CLASE — {N_REPEATS}x{N_FOLDS}-Fold CV")
    print(f"  Distante GLCM: {DISTANCES}")
    print(f"  Nr. parametri: {len(FEATURE_COLUMNS)}")
    print(f"{'='*60}\n")

    if not CSV_PATH.exists():
        print(f"[EROARE] Nu gasesc CSV-ul la: {CSV_PATH}")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)

    df["section_norm"] = df["section"].astype(str).str[0]
    df["probe_base"]   = df["probe"].astype(str).str.replace("HE", "", regex=False)
    df["key"]          = df["probe_base"] + "_" + df["section_norm"]
    df["diagnosis"]    = df["key"].map(DIAGNOSIS)

    df = df.dropna(subset=["diagnosis"])

    print(f"  Total imagini incarcate : {len(df)}")
    for c in CLASSES:
        n_img   = (df["diagnosis"] == c).sum()
        n_group = df.loc[df["diagnosis"] == c, "key"].nunique()
        print(f"  {c.capitalize():<10}: {n_img:>4} imagini | {n_group} grupuri de tesut")

    return df


# ──────────────────────────── CROSS-VALIDATION ──────────────────────────────

def cross_validate(df):
    probe_sections = (
        df.groupby("key")["diagnosis"]
        .agg(lambda x: x.value_counts().index[0])
        .reset_index()
        .rename(columns={"diagnosis": "dominant_class"})
    )

    keys    = probe_sections["key"].values
    classes = probe_sections["dominant_class"].values

    print(f"\n  Sectiuni unice : {len(keys)}")
    for c in CLASSES:
        print(f"  {c.capitalize():<10}: {(classes == c).sum()}")
    print(f"  Repetari       : {N_REPEATS}")
    print(f"  Fold-uri       : {N_FOLDS}")
    print(f"  Total evaluari : {N_REPEATS * N_FOLDS}\n")

    acc_results   = []
    bal_acc_results = []
    all_y_true    = []
    all_y_pred    = []

    for repeat in range(N_REPEATS):
        skf = StratifiedKFold(
            n_splits=N_FOLDS,
            shuffle=True,
            random_state=RANDOM_STATE + repeat
        )

        for train_idx, test_idx in skf.split(keys, classes):
            train_keys = keys[train_idx]
            test_keys  = keys[test_idx]

            train_df = df[df["key"].isin(train_keys)]
            test_df  = df[df["key"].isin(test_keys)]

            if train_df["diagnosis"].nunique() < 3 or len(test_df) == 0:
                continue

            X_train = train_df[FEATURE_COLUMNS].astype(float)
            y_train = train_df["diagnosis"]
            X_test  = test_df[FEATURE_COLUMNS].astype(float)
            y_test  = test_df["diagnosis"]

            scaler  = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test  = scaler.transform(X_test)

            model = SVC(
                kernel=KERNEL,
                class_weight="balanced",
                random_state=RANDOM_STATE
            )
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            acc_results.append(accuracy_score(y_test, y_pred))
            bal_acc_results.append(balanced_accuracy_score(y_test, y_pred))
            all_y_true.extend(y_test.tolist())
            all_y_pred.extend(y_pred.tolist())

        if (repeat + 1) % 5 == 0:
            print(f"  Repetarea {repeat+1:2d}/{N_REPEATS} "
                  f"— accuracy: {np.mean(acc_results)*100:.1f}% "
                  f"| balanced accuracy: {np.mean(bal_acc_results)*100:.1f}%")

    return acc_results, bal_acc_results, all_y_true, all_y_pred


# ──────────────────────────── VIZUALIZARI ───────────────────────────────────

def plot_confusion_matrix(y_true, y_pred):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    cm   = confusion_matrix(y_true, y_pred, labels=CLASSES)
    disp = ConfusionMatrixDisplay(cm, display_labels=[c.capitalize() for c in CLASSES])
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — SVM {KERNEL} 3-clase\n"
                 f"d{'+d'.join(str(d) for d in DISTANCES)}", fontsize=13)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "confusion_matrix_3class.png", dpi=150)
    print(f"  Confusion matrix salvata")


def plot_accuracy_comparison(acc_results, bal_acc_results):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    means = [np.mean(acc_results) * 100, np.mean(bal_acc_results) * 100]
    stds  = [np.std(acc_results) * 100, np.std(bal_acc_results) * 100]
    labels = ["Accuracy\n(bruta)", "Balanced Accuracy\n(medie per clasa)"]

    bars = ax.bar(labels, means, yerr=stds, capsize=6,
                   color=["steelblue", "darkorange"])
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=13, fontweight="bold")

    ax.axhline(100 / 3, color="red", linestyle="--", linewidth=1.2,
               label="Random (33.3%, 3 clase)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"SVM rbf 3-clase — Accuracy vs Balanced Accuracy\n"
                 f"{N_REPEATS}x{N_FOLDS}-Fold CV")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "accuracy_vs_balanced_3class.png", dpi=150)
    print(f"  Grafic accuracy vs balanced accuracy salvat")


def plot_feature_importance(df):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    X = df[FEATURE_COLUMNS].astype(float)
    y = df["diagnosis"]

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = SVC(kernel=KERNEL, class_weight="balanced", random_state=RANDOM_STATE)
    model.fit(X_scaled, y)

    print(f"\n  Calculare Permutation Importance (3-clase)...")
    perm = permutation_importance(
        model, X_scaled, y,
        n_repeats=10,
        random_state=RANDOM_STATE,
        scoring="balanced_accuracy"
    )

    indices = np.argsort(perm.importances_mean)[::-1]
    top_n   = min(20, len(FEATURE_COLUMNS))
    top_idx = indices[:top_n]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(top_n), perm.importances_mean[top_idx],
           yerr=perm.importances_std[top_idx], color="steelblue", capsize=3)
    ax.set_xticks(range(top_n))
    ax.set_xticklabels([FEATURE_COLUMNS[i] for i in top_idx],
                        rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Scadere medie balanced accuracy")
    ax.set_title(f"Permutation Importance — SVM {KERNEL} 3-clase (Top 20)", fontsize=13)
    ax.axhline(0, color="red", linestyle="--", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "feature_importance_3class.png", dpi=150)
    print(f"  Feature importance salvata")

    print(f"\n  Top 10 parametri (dupa balanced accuracy):")
    print(f"  {'Parametru':<35} {'Importanta':>12} {'Std':>8}")
    print(f"  {'-'*57}")
    for i in indices[:10]:
        print(f"  {FEATURE_COLUMNS[i]:<35} {perm.importances_mean[i]:>12.4f} {perm.importances_std[i]:>8.4f}")


# ──────────────────────────── ENTRY POINT ───────────────────────────────────

if __name__ == "__main__":
    df = load_data()

    acc_results, bal_acc_results, all_y_true, all_y_pred = cross_validate(df)

    mean_acc     = np.mean(acc_results)
    mean_bal_acc = np.mean(bal_acc_results)

    print(f"\n{'='*60}")
    print(f"  Rezultate finale — SVM {KERNEL} 3-clase (d{'+d'.join(str(d) for d in DISTANCES)}):")
    print(f"{'='*60}")
    print(f"  Accuracy medie (bruta)    : {mean_acc*100:.1f}% (+/- {np.std(acc_results)*100:.1f}%)")
    print(f"  Balanced accuracy medie   : {mean_bal_acc*100:.1f}% (+/- {np.std(bal_acc_results)*100:.1f}%)")
    print(f"  (balanced accuracy = media accuracy-ului per clasa, corecteaza")
    print(f"   dezechilibrul: 9 normal / 9 malign / 5 displazic)")

    print(f"\n  Raport detaliat (agregat pe toate cele {N_REPEATS*N_FOLDS} evaluari):")
    print(classification_report(
        all_y_true, all_y_pred,
        labels=CLASSES,
        zero_division=0
    ))

    print(f"\n  Generare grafice...")
    plot_confusion_matrix(all_y_true, all_y_pred)
    plot_accuracy_comparison(acc_results, bal_acc_results)
    plot_feature_importance(df)

    plt.close("all")

    print(f"\n{'='*60}")
    print(f"  Gata! Graficele sunt in: {PLOTS_DIR}")
    print(f"{'='*60}\n")