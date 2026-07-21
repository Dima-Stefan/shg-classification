"""
classify_svm_3class_balanced.py
Clasificare SVM in 3 clase (normal/malign/displazic) cu pool PERFECT
BALANSAT: la fiecare repetare, se sub-esantioneaza aleator exact 5
grupuri de tesut din normal (din cele 9 disponibile) si 5 din malign
(din cele 9 disponibile), pastrand toate cele 5 grupuri displazice --
rezultand un pool de 15 grupuri (5+5+5) pentru acea repetare.

Motivatie: class_weight='balanced' compenseaza dezechilibrul in functia
de cost a SVM, dar NU elimina asimetria de informatie -- clasele
majoritare tot ofera SVM-ului mai multe exemple pentru a invata granita
de decizie. Subesantionarea la 5/5/5 elimina complet acest avantaj
structural, oferind o comparatie mult mai directa intre cele 3 clase.

Fiecare din cele N_REPEATS repetari foloseste un subesantion DIFERIT
de 5 normale + 5 maligne (random_state diferit), pentru a nu depinde
de o singura alegere arbitrara -- rezultatul e mediat peste multe
subesantionari posibile.

Utilizare:
    python classify_svm_3class_balanced.py --csv "...\\features.csv"
    python classify_svm_3class_balanced.py --csv "...\\features_glrlm.csv"
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
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
)

# ──────────────────────────── CONFIGURARE FIXA ──────────────────────────────

DIAGNOSIS = {
    "72066_1" : "normal",  "72066_2" : "malign",  "72066_3" : "normal",
    "83064_1" : "normal",  "83064_2" : "malign",  "83064_3" : "malign",
    "89376_1" : "malign",  "89376_2" : "malign",  "89376_3" : "displazic",
    "102312_1": "normal",  "102312_2": "malign",  "102312_3": "malign",
    "149113_1": "displazic", "149113_2": "displazic", "149113_3": "displazic",
    "155820_1": "malign",  "155820_2": "malign",  "155820_3": "displazic",
    "156222_1": "normal",  "156222_2": "normal",  "156222_3": "normal",
    "182467_1": "normal",  "182467_3": "normal",
}

CLASSES      = ["normal", "malign", "displazic"]
KERNEL       = "rbf"
RANDOM_STATE = 42
N_FOLDS      = 3

EXCLUDED_COLUMNS = {
    "probe", "section", "polar", "channel", "is_HE", "filepath",
    "patch_row", "patch_col",
    "section_norm", "probe_base", "key", "diagnosis", "group_id",
    "dominant_class", "polar_num",
}


def detect_feature_columns(df):
    candidate_columns = [c for c in df.columns if c not in EXCLUDED_COLUMNS]
    feature_columns = []
    for col in candidate_columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().mean() >= 0.95:
            feature_columns.append(col)
    return feature_columns


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

    feature_columns = detect_feature_columns(df)
    for col in feature_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=feature_columns)

    print(f"\n  CSV incarcat: {csv_path.name}")
    group_by_class = {}
    for c in CLASSES:
        groups = sorted(df.loc[df["diagnosis"] == c, "key"].unique())
        group_by_class[c] = groups
        print(f"  {c.capitalize():<10}: {len(groups)} grupuri de tesut -> {groups}")
    print(f"  Parametri detectati: {len(feature_columns)}")

    min_groups = min(len(g) for g in group_by_class.values())
    print(f"\n  Pool balansat: {min_groups} grupuri per clasa "
          f"(limitat de clasa minoritara: displazic)")

    return df, feature_columns, group_by_class, min_groups


# ──────────────────────────── CROSS-VALIDATION BALANSATA ────────────────────

def cross_validate_balanced(df, feature_columns, group_by_class, n_per_class, n_folds, n_repeats):
    """
    La fiecare repetare: sub-esantioneaza aleator n_per_class grupuri din
    fiecare clasa (folosind un RandomState propriu per repetare), apoi
    ruleaza StratifiedKFold pe acest pool balansat.
    """
    acc_results, bal_acc_results = [], []
    all_y_true, all_y_pred = [], []
    subsample_logs = []

    for repeat in range(n_repeats):
        rng = np.random.RandomState(RANDOM_STATE + repeat)

        # Sub-esantionare balansata pentru aceasta repetare
        pool_keys, pool_classes = [], []
        for c in CLASSES:
            available = group_by_class[c]
            chosen = rng.choice(available, size=n_per_class, replace=False)
            pool_keys.extend(chosen)
            pool_classes.extend([c] * n_per_class)
        pool_keys    = np.array(pool_keys)
        pool_classes = np.array(pool_classes)
        subsample_logs.append(dict(zip(pool_keys, pool_classes)))

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                               random_state=RANDOM_STATE + repeat)

        for train_idx, test_idx in skf.split(pool_keys, pool_classes):
            train_keys = pool_keys[train_idx]
            test_keys  = pool_keys[test_idx]

            train_df = df[df["key"].isin(train_keys)]
            test_df  = df[df["key"].isin(test_keys)]

            if train_df["diagnosis"].nunique() < 3 or len(test_df) == 0:
                continue

            X_train = train_df[feature_columns].astype(float)
            y_train = train_df["diagnosis"]
            X_test  = test_df[feature_columns].astype(float)
            y_test  = test_df["diagnosis"]

            scaler  = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test  = scaler.transform(X_test)

            # class_weight='balanced' e practic un no-op aici (clasele
            # sunt deja egale ca numar de grupuri), dar il pastram pentru
            # ca numarul de IMAGINI per grup difera usor (720/700/400 total)
            model = SVC(kernel=KERNEL, class_weight="balanced", random_state=RANDOM_STATE)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc_results.append(accuracy_score(y_test, y_pred))
            bal_acc_results.append(balanced_accuracy_score(y_test, y_pred))
            all_y_true.extend(y_test.tolist())
            all_y_pred.extend(y_pred.tolist())

        if (repeat + 1) % 5 == 0:
            print(f"  Repetarea {repeat+1:2d}/{n_repeats} "
                  f"— acc: {np.mean(acc_results)*100:.1f}% "
                  f"| bal.acc: {np.mean(bal_acc_results)*100:.1f}%")

    return acc_results, bal_acc_results, all_y_true, all_y_pred, subsample_logs


# ──────────────────────────── VIZUALIZARI ───────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, csv_name, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    cm   = confusion_matrix(y_true, y_pred, labels=CLASSES)
    disp = ConfusionMatrixDisplay(cm, display_labels=[c.capitalize() for c in CLASSES])
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — SVM {KERNEL} 3-clase BALANSAT (5/5/5) — {csv_name}",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(output_dir / f"confusion_matrix_3class_balanced_{csv_name}.png", dpi=150)
    print(f"  Confusion matrix salvata")


def plot_balanced_vs_unbalanced(acc_bal, bal_acc_bal, csv_name, n_per_class, output_dir, n_folds, n_repeats):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    means = [np.mean(acc_bal) * 100, np.mean(bal_acc_bal) * 100]
    stds  = [np.std(acc_bal) * 100, np.std(bal_acc_bal) * 100]
    labels = ["Accuracy\n(bruta)", "Balanced Accuracy\n(medie per clasa)"]

    bars = ax.bar(labels, means, yerr=stds, capsize=6, color=["mediumseagreen", "seagreen"])
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=13, fontweight="bold")

    ax.axhline(100 / 3, color="red", linestyle="--", linewidth=1.2, label="Random (33.3%)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"SVM rbf 3-clase — Pool BALANSAT ({n_per_class}/{n_per_class}/{n_per_class}) — {csv_name}\n"
                 f"{n_repeats}x{n_folds}-Fold CV, sub-esantionare diferita per repetare")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"accuracy_balanced_pool_{csv_name}.png", dpi=150)
    print(f"  Grafic accuracy pool balansat salvat")


# ──────────────────────────── ENTRY POINT ───────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Clasificare SVM 3-clase cu pool balansat prin sub-esantionare"
    )
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--n-repeats", type=int, default=25)
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    csv_name   = args.csv.stem
    output_dir = args.output_dir or (args.csv.parent.parent / "Plots" / f"svm_3class_balanced_{csv_name}")

    print(f"\n{'='*60}")
    print(f"  Clasificare SVM 3-clase — Pool BALANSAT")
    print(f"  CSV: {args.csv}")
    print(f"{'='*60}")

    df, feature_columns, group_by_class, n_per_class = load_data(args.csv)

    print(f"\n{'='*60}")
    print(f"  SVM rbf — pool {n_per_class}/{n_per_class}/{n_per_class}, "
          f"{args.n_repeats}x{args.n_folds}-Fold CV")
    print(f"{'='*60}")

    acc_results, bal_acc_results, all_y_true, all_y_pred, subsample_logs = cross_validate_balanced(
        df, feature_columns, group_by_class, n_per_class, args.n_folds, args.n_repeats
    )

    mean_acc     = np.mean(acc_results)
    mean_bal_acc = np.mean(bal_acc_results)

    print(f"\n{'='*60}")
    print(f"  REZULTATE FINALE — {csv_name} — pool balansat {n_per_class}/{n_per_class}/{n_per_class}")
    print(f"{'='*60}")
    print(f"  Accuracy medie (bruta)  : {mean_acc*100:.1f}% (+/- {np.std(acc_results)*100:.1f}%)")
    print(f"  Balanced accuracy medie : {mean_bal_acc*100:.1f}% (+/- {np.std(bal_acc_results)*100:.1f}%)")
    print(f"  (Cu pool perfect balansat, accuracy bruta si balanced accuracy")
    print(f"   ar trebui sa fie foarte apropiate -- daca difera mult, verificati")
    print(f"   distributia claselor in fold-uri individuale)")

    print(f"\n  Raport detaliat (agregat pe toate evaluarile):")
    print(classification_report(all_y_true, all_y_pred, labels=CLASSES, zero_division=0))

    print(f"\n  Generare grafice...")
    plot_confusion_matrix(all_y_true, all_y_pred, csv_name, output_dir)
    plot_balanced_vs_unbalanced(acc_results, bal_acc_results, csv_name, n_per_class,
                                 output_dir, args.n_folds, args.n_repeats)

    plt.close("all")

    print(f"\n{'='*60}")
    print(f"  Gata! Graficele sunt in: {output_dir}")
    print(f"{'='*60}\n")