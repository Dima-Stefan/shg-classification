"""
classify_rf_kfold.py
Clasificarea imaginilor SHG (512x512) folosind Random Forest,
cu Stratified 3-Fold Cross-Validation repetat 25 ori -- metodologie
IDENTICA cu classify_svm_kfold.py, pentru a permite o comparatie
directa si corecta intre SVM si Random Forest ca algoritm.

Configuratie standard: distante GLCM d1 + d10 + d30 (aceleasi 64
parametri folositi ca baseline pentru SVM, 69.4% accuracy).

Nota tehnica: spre deosebire de SVM, Random Forest e invariant la
scala parametrilor (arborii de decizie aleg praguri pe fiecare
parametru independent, deci StandardScaler nu are efect asupra
performantei). Scaler-ul e pastrat totusi in pipeline pentru
consistenta cu restul scripturilor si pentru a nu introduce o
diferenta de procesare intre cele doua metode comparate.

Utilizare:
    python classify_rf_kfold.py
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
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
    PLOTS_DIR = PLOTS_DIR / "rf_kfold"
except ImportError:
    CSV_PATH  = Path(r"C:\Users\dimas\Downloads\projects\venv\Results\CSV\features.csv")
    PLOTS_DIR = Path(r"C:\Users\dimas\Downloads\projects\venv\Results\Plots\rf_kfold")
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
RANDOM_STATE = 42

# Hiperparametri Random Forest -- valori implicite rezonabile pentru un
# set de date mic (nu s-a facut Grid Search, la fel ca alegerea kernelului
# SVM initial; poate fi rafinat ulterior daca rezultatele o justifica)
N_ESTIMATORS = 500
MAX_DEPTH    = None   # arborii cresc pana la frunze pure (RF compenseaza prin ansamblu)


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
    print(f"  Random Forest — {N_REPEATS}x{N_FOLDS}-Fold CV")
    print(f"  Distante GLCM: {DISTANCES}")
    print(f"  Nr. parametri: {len(FEATURE_COLUMNS)}")
    print(f"  N_ESTIMATORS : {N_ESTIMATORS}")
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

            model = RandomForestClassifier(
                n_estimators=N_ESTIMATORS,
                max_depth=MAX_DEPTH,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
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
    ax.bar(range(1, len(results) + 1), [r * 100 for r in results], color="darkseagreen")
    ax.axhline(np.mean(results) * 100, color="red", linestyle="--",
               label=f"Medie: {np.mean(results)*100:.1f}%")
    ax.set_xlabel("Evaluare")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Accuracy per evaluare — Random Forest d{'+d'.join(str(d) for d in DISTANCES)}\n"
                 f"({N_REPEATS}x{N_FOLDS}-Fold CV)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "cv_accuracy.png", dpi=150)
    print(f"  CV accuracy salvat")


def plot_confusion_matrix(y_true, y_pred):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    cm   = confusion_matrix(y_true, y_pred, labels=["normal", "malign"])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Malign"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Greens")
    ax.set_title(f"Confusion Matrix — Random Forest d{'+d'.join(str(d) for d in DISTANCES)}", fontsize=13)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "confusion_matrix.png", dpi=150)
    print(f"  Confusion matrix salvata")


def plot_feature_importance(df):
    """
    Doua tipuri de importanta, calculate pe modelul final antrenat pe
    tot setul de date:
      1. Feature importance NATIVA a Random Forest (Mean Decrease in
         Impurity -- cat de mult scade impuritatea Gini la fiecare split
         care foloseste acel parametru, mediat pe toti arborii).
      2. Permutation importance (aceeasi metoda folosita si pentru SVM,
         Sectiunea 6.5), pentru comparabilitate directa intre metode.
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    X = df[FEATURE_COLUMNS].astype(float)
    y = df["diagnosis"]

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    )
    model.fit(X_scaled, y)

    # --- 1. Native feature importance (Mean Decrease in Impurity) ---
    native_importances = model.feature_importances_
    indices_native = np.argsort(native_importances)[::-1]
    top_n = min(20, len(FEATURE_COLUMNS))
    top_idx_native = indices_native[:top_n]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(top_n), native_importances[top_idx_native], color="darkseagreen")
    ax.set_xticks(range(top_n))
    ax.set_xticklabels([FEATURE_COLUMNS[i] for i in top_idx_native],
                        rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean Decrease in Impurity (Gini)")
    ax.set_title("Feature Importance NATIVA — Random Forest (Top 20 parametri)", fontsize=13)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "feature_importance_native.png", dpi=150)
    print(f"  Feature importance nativa (MDI) salvata")

    # --- 2. Permutation importance (comparabil cu SVM) ---
    print(f"\n  Calculare Permutation Importance...")
    perm = permutation_importance(
        model, X_scaled, y,
        n_repeats=10,
        random_state=RANDOM_STATE,
        scoring="accuracy"
    )

    indices_perm = np.argsort(perm.importances_mean)[::-1]
    top_idx_perm = indices_perm[:top_n]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(
        range(top_n),
        perm.importances_mean[top_idx_perm],
        yerr=perm.importances_std[top_idx_perm],
        color="mediumseagreen", capsize=3
    )
    ax.set_xticks(range(top_n))
    ax.set_xticklabels([FEATURE_COLUMNS[i] for i in top_idx_perm],
                        rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Scadere medie accuracy")
    ax.set_title("Permutation Importance — Random Forest (Top 20 parametri)", fontsize=13)
    ax.axhline(0, color="red", linestyle="--", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "feature_importance_permutation.png", dpi=150)
    print(f"  Permutation importance salvata")

    print(f"\n  Top 10 parametri (Feature Importance NATIVA - MDI):")
    print(f"  {'Parametru':<35} {'MDI':>10}")
    print(f"  {'-'*47}")
    for i in indices_native[:10]:
        print(f"  {FEATURE_COLUMNS[i]:<35} {native_importances[i]:>10.4f}")

    print(f"\n  Top 10 parametri (Permutation Importance):")
    print(f"  {'Parametru':<35} {'Importanta':>12} {'Std':>8}")
    print(f"  {'-'*57}")
    for i in indices_perm[:10]:
        print(f"  {FEATURE_COLUMNS[i]:<35} {perm.importances_mean[i]:>12.4f} {perm.importances_std[i]:>8.4f}")


# ──────────────────────────── ENTRY POINT ───────────────────────────────────

if __name__ == "__main__":
    df = load_data()

    results, all_y_true, all_y_pred = cross_validate(df)

    mean_acc = np.mean(results)
    std_acc  = np.std(results)

    print(f"\n{'='*60}")
    print(f"  Rezultate finale — Random Forest (d{'+d'.join(str(d) for d in DISTANCES)}):")
    print(f"{'='*60}")
    print(f"  Accuracy medie  : {mean_acc*100:.1f}% (+/- {std_acc*100:.1f}%)")
    print(f"  Accuracy minima : {min(results)*100:.1f}%")
    print(f"  Accuracy maxima : {max(results)*100:.1f}%")
    print(f"\n  Referinta SVM rbf (aceeasi metodologie): 69.4%")
    print(f"\n  Raport detaliat:")
    print(classification_report(
        all_y_true, all_y_pred,
        labels=["normal", "malign"],
        zero_division=0
    ))

    print(f"\n  Generare grafice...")
    plot_cv_accuracy(results)
    plot_confusion_matrix(all_y_true, all_y_pred)
    plot_feature_importance(df)

    plt.close("all")

    print(f"\n{'='*60}")
    print(f"  Gata! Graficele sunt in: {PLOTS_DIR}")
    print(f"{'='*60}\n")