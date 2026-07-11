"""
classify_svm_kfold.py
Clasificarea imaginilor SHG (512x512) folosind SVM cu kernel rbf
cu Stratified 3-Fold Cross-Validation repetat 25 ori.

Eliminare data leakage prin gruparea sectiunilor pereche:
  Sectiunea 1  <-> Sectiunea 11  (aceeasi zona, sectiuni consecutive)
  Sectiunea 2  <-> Sectiunea 22
  Sectiunea 3  <-> Sectiunea 33

Un grup de tesut contine toate 4 sectiuni pereche:
  [probe_1, probe_11, probeHE_1, probeHE_11]

Split-ul se face la nivel de GRUP — astfel toate sectiunile
din aceeasi zona de tesut sunt fie in train, fie in test.

Testeaza 4 scenarii HE vs non-HE:
  1. non-HE → non-HE
  2. HE → HE
  3. non-HE → HE
  4. HE → non-HE

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
PLOTS_DIR = _PLOTS_BASE / "svm_groups"

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
    "182467_3": "normal",
}

# Maparea sectiunilor pereche (aceeasi zona de tesut)
SECTION_PAIRS = {
    "1": "11",
    "2": "22",
    "3": "33",
    "11": "1",
    "22": "2",
    "33": "3",
}

DIRECTION_PAIRS = ["0_180", "90_270", "45_225", "135_315"]
GLCM_PARAMS     = ["ASM", "Contrast", "Correlation", "IDM", "Entropy"]
DISTANCES       = [1, 10, 30]

N_FOLDS      = 3
N_REPEATS    = 25
KERNEL       = "rbf"
RANDOM_STATE = 42

FEATURE_COLUMNS = [
    "Mean", "StDev", "Skew", "Kurt",
] + [
    f"{param}_{pair}_d{dist}"
    for pair in DIRECTION_PAIRS
    for dist in DISTANCES
    for param in GLCM_PARAMS
]


# ──────────────────────────── INCARCARE DATE ────────────────────────────────

def load_data():
    print(f"\n{'='*60}")
    print(f"  SVM {KERNEL} — {N_REPEATS}x{N_FOLDS}-Fold CV")
    print(f"  Split la nivel de grup de tesut (fara data leakage)")
    print(f"  Distante GLCM: {DISTANCES}")
    print(f"  Nr. parametri: {len(FEATURE_COLUMNS)}")
    print(f"{'='*60}\n")

    if not CSV_PATH.exists():
        print(f"[EROARE] Nu gasesc CSV-ul la: {CSV_PATH}")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)

    df["section_norm"] = df["section"].astype(str).str[0]
    df["probe_base"]   = df["probe"].str.replace("HE", "", regex=False)
    df["key"]          = df["probe_base"] + "_" + df["section_norm"]
    df["diagnosis"]    = df["key"].map(DIAGNOSIS)

    # Adaugam coloana group_id — identifica zona de tesut
    # Sectiunile 1 si 11 sunt aceeasi zona -> group_id = probe_base + "_g1"
    # Sectiunile 2 si 22 sunt aceeasi zona -> group_id = probe_base + "_g2"
    # Sectiunile 3 si 33 sunt aceeasi zona -> group_id = probe_base + "_g3"
    def get_group(row):
        s = str(row["section"])
        base = row["probe_base"]
        if s in ["1", "11"]:
            return f"{base}_g1"
        elif s in ["2", "22"]:
            return f"{base}_g2"
        elif s in ["3", "33"]:
            return f"{base}_g3"
        return f"{base}_g{s}"

    df["group_id"] = df.apply(get_group, axis=1)

    df = df.dropna(subset=["diagnosis"])
    df = df[df["diagnosis"] != "displazic"]

    non_he = df[df["is_HE"] == False].copy()
    he     = df[df["is_HE"] == True].copy()

    # Grupurile unice
    all_groups = sorted(df["group_id"].unique())
    print(f"  Total imagini (fara displazic): {len(df)}")
    print(f"  Non-HE : {len(non_he)} imagini")
    print(f"  HE     : {len(he)} imagini")
    print(f"  Grupuri unice de tesut: {len(all_groups)}")
    for g in all_groups:
        diag = df[df["group_id"]==g]["diagnosis"].iloc[0]
        print(f"    {g:<20} → {diag}")

    return df, non_he, he


# ──────────────────────────── CROSS-VALIDATION ──────────────────────────────

def get_groups(df):
    """
    Returneaza grupurile unice cu clasa dominanta.
    Split-ul se face la nivel de grup.
    """
    groups = (
        df.groupby("group_id")["diagnosis"]
        .agg(lambda x: x.value_counts().index[0])
        .reset_index()
        .rename(columns={"diagnosis": "dominant_class"})
    )
    return groups["group_id"].values, groups["dominant_class"].values


def cross_validate(df_train_pool, df_test_pool, label):
    """
    CV la nivel de grup de tesut.

    La fiecare iteratie:
      - 3-fold pe grupurile din TEST pool → 1 fold = TEST
      - 3-fold pe grupurile din TRAIN pool → 2 fold-uri = TRAIN

    Grupurile din test si train nu se suprapun niciodata,
    eliminand data leakage intre sectiunile pereche.
    """
    train_groups, train_classes = get_groups(df_train_pool)
    test_groups,  test_classes  = get_groups(df_test_pool)

    same_pool = (df_train_pool is df_test_pool)

    print(f"\n  [{label}]")
    print(f"  Train pool: {len(train_groups)} grupuri | Test pool: {len(test_groups)} grupuri")

    results    = []
    all_y_true = []
    all_y_pred = []

    for repeat in range(N_REPEATS):
        skf_test = StratifiedKFold(
            n_splits=N_FOLDS,
            shuffle=True,
            random_state=RANDOM_STATE + repeat
        )
        skf_train = StratifiedKFold(
            n_splits=N_FOLDS,
            shuffle=True,
            random_state=RANDOM_STATE + repeat + 1000
        )

        test_splits  = list(skf_test.split(test_groups, test_classes))
        train_splits = list(skf_train.split(train_groups, train_classes))

        for fold in range(N_FOLDS):
            # TEST: fold-ul curent din test pool
            _, test_idx      = test_splits[fold]
            fold_test_groups = test_groups[test_idx]

            # TRAIN: cele 2 fold-uri ramase din train pool
            train_idx, _      = train_splits[fold]
            fold_train_groups = train_groups[train_idx]

            if same_pool:
                # Eliminam din train grupurile care sunt in test
                fold_train_groups = np.setdiff1d(
                    fold_train_groups, fold_test_groups
                )

            # Filtram datele pe baza grupurilor
            train_df = df_train_pool[
                df_train_pool["group_id"].isin(fold_train_groups)
            ]
            test_df = df_test_pool[
                df_test_pool["group_id"].isin(fold_test_groups)
            ]

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
            print(f"  Repetarea {repeat+1:2d}/{N_REPEATS} "
                  f"— acc medie: {np.mean(results)*100:.1f}%")

    mean_acc = np.mean(results)
    std_acc  = np.std(results)
    print(f"  Accuracy medie : {mean_acc*100:.1f}% (+/- {std_acc*100:.1f}%)")
    print(f"  Accuracy minima: {min(results)*100:.1f}%")
    print(f"  Accuracy maxima: {max(results)*100:.1f}%")
    print(classification_report(
        all_y_true, all_y_pred,
        labels=["normal", "malign"],
        zero_division=0
    ))

    return results, all_y_true, all_y_pred


# ──────────────────────────── VIZUALIZARI ───────────────────────────────────

def plot_scenario_comparison(scenario_results):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    names  = list(scenario_results.keys())
    means  = [np.mean(scenario_results[k]) * 100 for k in names]
    colors = ["steelblue", "darkorange", "green", "purple"]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(names, means, color=colors, width=0.5)
    ax.axhline(50,   color="red",  linestyle="--", linewidth=1.5,
               label="Random (50%)")
    ax.axhline(69.4, color="gray", linestyle=":",  linewidth=1.5,
               label="Referinta anterioara (69.4%)")

    for bar, val in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}%",
            ha="center", va="bottom", fontsize=12, fontweight="bold"
        )

    ax.set_ylabel("Accuracy medie (%)", fontsize=12)
    ax.set_title(
        f"Comparatie scenarii HE vs non-HE — SVM {KERNEL}\n"
        f"({N_REPEATS}x{N_FOLDS}-Fold CV la nivel de grup de tesut\n"
        f"Fara data leakage intre sectiuni pereche)",
        fontsize=12
    )
    ax.set_ylim(0, 100)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "scenario_comparison.png", dpi=150)
    print(f"  Grafic comparatie scenarii salvat")


def plot_confusion_matrix(y_true, y_pred, scenario_name):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    cm   = confusion_matrix(y_true, y_pred, labels=["normal", "malign"])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Malign"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {scenario_name}", fontsize=12)
    plt.tight_layout()
    safe = scenario_name.replace(" ", "_").replace("→", "to")
    plt.savefig(PLOTS_DIR / f"confusion_matrix_{safe}.png", dpi=150)


# ──────────────────────────── ENTRY POINT ───────────────────────────────────

if __name__ == "__main__":
    df, non_he, he = load_data()

    scenario_results = {}

    # ── Scenariul 1: non-HE → non-HE ────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Scenariul 1: non-HE → non-HE")
    print(f"{'='*60}")
    r1, yt1, yp1 = cross_validate(non_he, non_he, "non-HE → non-HE")
    scenario_results["non-HE\n→ non-HE"] = r1
    plot_confusion_matrix(yt1, yp1, "non-HE → non-HE")

    # ── Scenariul 2: HE → HE ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Scenariul 2: HE → HE")
    print(f"{'='*60}")
    r2, yt2, yp2 = cross_validate(he, he, "HE → HE")
    scenario_results["HE\n→ HE"] = r2
    plot_confusion_matrix(yt2, yp2, "HE → HE")

    # ── Scenariul 3: non-HE → HE ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Scenariul 3: non-HE → HE")
    print(f"{'='*60}")
    r3, yt3, yp3 = cross_validate(non_he, he, "non-HE → HE")
    scenario_results["non-HE\n→ HE"] = r3
    plot_confusion_matrix(yt3, yp3, "non-HE → HE")

    # ── Scenariul 4: HE → non-HE ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Scenariul 4: HE → non-HE")
    print(f"{'='*60}")
    r4, yt4, yp4 = cross_validate(he, non_he, "HE → non-HE")
    scenario_results["HE\n→ non-HE"] = r4
    plot_confusion_matrix(yt4, yp4, "HE → non-HE")

    # ── Sumar final ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Sumar scenarii (CV la nivel de grup, fara data leakage):")
    print(f"{'='*60}")
    print(f"  {'Scenariu':<25} {'Acc medie':>10} {'Acc min':>8} {'Acc max':>8}")
    print(f"  {'-'*55}")
    for name, results in zip(
        ["non-HE → non-HE", "HE → HE", "non-HE → HE", "HE → non-HE"],
        [r1, r2, r3, r4]
    ):
        print(f"  {name:<25} {np.mean(results)*100:>9.1f}%"
              f" {min(results)*100:>7.1f}% {max(results)*100:>7.1f}%")
    print(f"  {'Referinta anterioara':<25} {'69.4%':>10}")

    print(f"\n  Generare grafice...")
    plot_scenario_comparison(scenario_results)

    plt.close("all")

    print(f"\n{'='*60}")
    print(f"  Gata! Graficele sunt in: {PLOTS_DIR}")
    print(f"{'='*60}\n")