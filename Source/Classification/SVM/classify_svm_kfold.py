"""
classify_svm_kfold.py
Clasificarea imaginilor SHG (512x512) folosind SVM cu kernel rbf
cu Stratified 3-Fold Cross-Validation repetat 25 ori.

Configuratie standard: distante GLCM d1 + d10 + d30
(cea mai buna combinatie identificata experimental)

Utilizare:
    python classify_svm_kfold.py
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
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)

# ──────────────────────────── CONFIGURARE ───────────────────────────────────

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "Source" / "core"))
from config import CSV_FEATURES as CSV_PATH, PLOTS_DIR as _PLOTS_BASE
PLOTS_DIR = _PLOTS_BASE / "svm_kfold"

DIAGNOSIS = {
    "72066_1" : "normal",
    "72066_2" : "malign",
    "72066_3" : "normal",

    "83064_1" : "normal",
    "83064_2" : "malign",
    "83064_3" : "malign",

    "89376_1" : "malign",
    "89376_2" : "malign",
    "89376_3" : "displazic",

    "102312_1": "normal",
    "102312_2": "malign",
    "102312_3": "malign",

    "149113_1": "displazic",
    "149113_2": "displazic",
    "149113_3": "displazic",

    "155820_1": "malign",
    "155820_2": "malign",
    "155820_3": "displazic",

    "156222_1": "normal",
    "156222_2": "normal",
    "156222_3": "normal",

    "182467_1": "normal",
    "182467_3": "normal",       #23 seturi - 5 displazice = 18 seturi normale/maligne(9/9)
}

# Perechile de directii
DIRECTION_PAIRS = ["0_180", "90_270", "45_225", "135_315"]

# Parametrii GLCM
GLCM_PARAMS = ["ASM", "Contrast", "Correlation", "IDM", "Entropy"]

# Configuratie standard — distantele optime identificate experimental
DISTANCES = [1, 10, 30]

# Parametri cross-validation
N_FOLDS      = 3
N_REPEATS    = 25
KERNEL       = "rbf"
RANDOM_STATE = 42


# ──────────────────────────── FEATURE COLUMNS ───────────────────────────────

def build_feature_columns(distances):
    """
    Construieste lista de coloane pentru distantele specificate.
    Include parametrii histograma + GLCM la distantele date.
    """
    return [
        "Mean", "StDev", "Skew", "Kurt",
    ] + [
        f"{param}_{pair}_d{dist}"
        for pair in DIRECTION_PAIRS
        for dist in distances
        for param in GLCM_PARAMS
    ]


# Coloanele folosite in aceasta rulare
FEATURE_COLUMNS = build_feature_columns(DISTANCES)


# ──────────────────────────── INCARCARE DATE ────────────────────────────────

def load_data():
    print(f"\n{'='*60}")
    print(f"  SVM {KERNEL} — {N_REPEATS}x{N_FOLDS}-Fold CV")
    print(f"  Distante GLCM: {DISTANCES}")
    print(f"  Nr. parametri: {len(FEATURE_COLUMNS)}")
    print(f"{'='*60}\n")

    if not CSV_PATH.exists():
        print(f"[EROARE] Nu gasesc CSV-ul la: {CSV_PATH}")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    print(f"  Total imagini incarcate : {len(df)}")

    df["section_norm"] = df["section"].astype(str).str[0]
    df["probe_base"]   = df["probe"].str.replace("HE", "", regex=False)
    df["key"]          = df["probe_base"] + "_" + df["section_norm"]
    df["diagnosis"]    = df["key"].map(DIAGNOSIS)

    df = df.dropna(subset=["diagnosis"])
    df = df[df["diagnosis"] != "displazic"]

    print(f"  Normal    : {(df['diagnosis']=='normal').sum()}")
    print(f"  Malign    : {(df['diagnosis']=='malign').sum()}")

    return df


# ──────────────────────────── CROSS-VALIDATION ──────────────────────────────

def cross_validate(df):
    """
    Stratified 3-Fold CV repetat 25 ori la nivel de proba+sectiune.

    La fiecare repetare cele 18 sectiuni sunt impartite in 3 fold-uri
    stratificate (echilibrat normal/malign), fiecare fold fiind folosit
    pe rand ca test set.
    """
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
    print(f"  Fold-uri       : {N_FOLDS}")
    print(f"  Total evaluari : {N_REPEATS * N_FOLDS}\n")

    results    = []
    all_y_true = []
    all_y_pred = []

    for repeat in range(N_REPEATS):
        skf = StratifiedKFold(
            n_splits=N_FOLDS,
            shuffle=True,
            random_state=RANDOM_STATE + repeat
        )

        for fold, (train_idx, test_idx) in enumerate(skf.split(keys, classes)):
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

            model = SVC(
                kernel=KERNEL,
                class_weight="balanced",
                random_state=RANDOM_STATE
            )
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            acc    = accuracy_score(y_test, y_pred)

            results.append(acc)
            all_y_true.extend(y_test.tolist())
            all_y_pred.extend(y_pred.tolist())

        if (repeat + 1) % 5 == 0:
            print(f"  Repetarea {repeat+1:2d}/{N_REPEATS} — accuracy medie: {np.mean(results)*100:.1f}%")

    return results, all_y_true, all_y_pred


# ──────────────────────────── VIZUALIZARI ───────────────────────────────────

def plot_cv_accuracy(results):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(1, len(results) + 1), [r * 100 for r in results], color="steelblue")
    ax.axhline(np.mean(results) * 100, color="red", linestyle="--",
               label=f"Medie: {np.mean(results)*100:.1f}%")
    ax.set_xlabel("Evaluare")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Accuracy per evaluare — SVM {KERNEL} d{'+d'.join(str(d) for d in DISTANCES)}\n({N_REPEATS}x{N_FOLDS}-Fold CV)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "cv_accuracy.png", dpi=150)
    print(f"  CV accuracy salvat")


def plot_repeat_accuracy(results):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    results_per_repeat = [
        np.mean(results[i*N_FOLDS:(i+1)*N_FOLDS]) * 100
        for i in range(N_REPEATS)
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, N_REPEATS + 1), results_per_repeat,
            marker="o", color="steelblue", linewidth=2)
    ax.axhline(np.mean(results_per_repeat), color="red", linestyle="--",
               label=f"Medie: {np.mean(results_per_repeat):.1f}%")
    ax.set_xlabel("Repetare")
    ax.set_ylabel("Accuracy medie (%)")
    ax.set_title(f"Accuracy medie per repetare — SVM {KERNEL}")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "accuracy_per_repeat.png", dpi=150)
    print(f"  Accuracy per repetare salvata")


def plot_confusion_matrix(y_true, y_pred):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    cm   = confusion_matrix(y_true, y_pred, labels=["normal", "malign"])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Malign"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — SVM {KERNEL} d{'+d'.join(str(d) for d in DISTANCES)}", fontsize=13)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "confusion_matrix.png", dpi=150)
    print(f"  Confusion matrix salvata")


def plot_feature_importance(df):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    X = df[FEATURE_COLUMNS].astype(float)
    y = df["diagnosis"]

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = SVC(kernel=KERNEL, class_weight="balanced", random_state=RANDOM_STATE)
    model.fit(X_scaled, y)

    print(f"\n  Calculare Permutation Importance...")
    perm = permutation_importance(
        model, X_scaled, y,
        n_repeats=10,
        random_state=RANDOM_STATE,
        scoring="accuracy"
    )

    indices = np.argsort(perm.importances_mean)[::-1]
    top_n   = min(20, len(FEATURE_COLUMNS))
    top_idx = indices[:top_n]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(
        range(top_n),
        perm.importances_mean[top_idx],
        yerr=perm.importances_std[top_idx],
        color="steelblue", capsize=3
    )
    ax.set_xticks(range(top_n))
    ax.set_xticklabels(
        [FEATURE_COLUMNS[i] for i in top_idx],
        rotation=45, ha="right", fontsize=8
    )
    ax.set_ylabel("Scadere medie accuracy")
    ax.set_title(f"Permutation Importance — SVM {KERNEL} (Top 20 parametri)", fontsize=13)
    ax.axhline(0, color="red", linestyle="--", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "feature_importance.png", dpi=150)
    print(f"  Feature importance salvata")

    print(f"\n  Top 10 parametri:")
    print(f"  {'Parametru':<35} {'Importanta':>12} {'Std':>8}")
    print(f"  {'-'*57}")
    for i in indices[:10]:
        print(f"  {FEATURE_COLUMNS[i]:<35} {perm.importances_mean[i]:>12.4f} {perm.importances_std[i]:>8.4f}")


# ──────────────────────────── ENTRY POINT ───────────────────────────────────

if __name__ == "__main__":
    df = load_data()

    results, all_y_true, all_y_pred = cross_validate(df)

    mean_acc = np.mean(results)
    std_acc  = np.std(results)

    print(f"\n{'='*60}")
    print(f"  Rezultate finale — SVM {KERNEL} (d{'+d'.join(str(d) for d in DISTANCES)}):")
    print(f"{'='*60}")
    print(f"  Accuracy medie  : {mean_acc*100:.1f}% (+/- {std_acc*100:.1f}%)")
    print(f"  Accuracy minima : {min(results)*100:.1f}%")
    print(f"  Accuracy maxima : {max(results)*100:.1f}%")
    print(f"\n  Raport detaliat:")
    print(classification_report(
        all_y_true, all_y_pred,
        labels=["normal", "malign"],
        zero_division=0
    ))

    print(f"\n  Generare grafice...")
    plot_cv_accuracy(results)
    plot_repeat_accuracy(results)
    plot_confusion_matrix(all_y_true, all_y_pred)
    plot_feature_importance(df)

    plt.close("all")

    print(f"\n{'='*60}")
    print(f"  Gata! Graficele sunt in: {PLOTS_DIR}")
    print(f"{'='*60}\n")