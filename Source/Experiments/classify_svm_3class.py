"""
classify_svm_3class.py
Clasificare SVM in 3 clase (normal/malign/displazic), generica si
reutilizabila pe orice CSV al proiectului (features.csv, features_glrlm.csv,
features_caf.csv, features_glcm_extended.csv etc.) -- consistent cu
select_features.py / select_features_rfe.py.

Coloanele de parametri sunt detectate AUTOMAT (toate coloanele numerice,
exceptand metadatele), deci nu trebuie specificate manual -- ruleaza
neschimbat indiferent daca CSV-ul are 64, 65, 84 sau alt numar de
parametri.

Metodologie identica cu classify_svm_kfold_3class.py: SVM rbf,
class_weight='balanced', StratifiedKFold 3-Fold repetat 25 ori la nivel
de grup de tesut (probe+sectiune), raportand atat accuracy bruta cat
si balanced accuracy (esentiala aici din cauza dezechilibrului de clase
9 normal / 9 malign / 5 displazic).

Utilizare:
    python classify_svm_3class.py --csv "...\\Results\\CSV\\features.csv"
    python classify_svm_3class.py --csv "...\\Results\\CSV\\features_glrlm.csv"
    python classify_svm_3class.py --csv "...\\Results\\CSV\\features_glrlm.csv" --group glrlm

Argumente:
    --csv         Calea catre CSV-ul de analizat (OBLIGATORIU)
    --group       Subsetul de parametri: all | glcm | glrlm (implicit: all)
    --n-repeats   Numarul de repetari ale CV (implicit 25)
    --n-folds     Numarul de fold-uri (implicit 3)
    --output-dir  Folderul pentru grafice (implicit langa CSV-ul de input)
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
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)

# ──────────────────────────── CONFIGURARE FIXA ──────────────────────────────

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

CLASSES      = ["normal", "malign", "displazic"]
KERNEL       = "rbf"
RANDOM_STATE = 42

EXCLUDED_COLUMNS = {
    "probe", "section", "polar", "channel", "is_HE", "filepath",
    "patch_row", "patch_col",
    "section_norm", "probe_base", "key", "diagnosis", "group_id",
    "dominant_class", "polar_num",
}

GLCM_PARAM_NAMES  = ["ASM", "Contrast", "Correlation", "IDM", "Entropy"]
GLRLM_PARAM_NAMES = ["SRE", "LRE", "GLN", "RLN", "RP"]
HISTOGRAM_NAMES   = ["Mean", "StDev", "Skew", "Kurt"]


def filter_columns_by_group(all_columns: list, group: str) -> list:
    if group == "all":
        return all_columns

    hist_cols = [c for c in all_columns if c in HISTOGRAM_NAMES]

    if group == "glcm":
        selected = hist_cols + [
            c for c in all_columns
            if any(c.startswith(p + "_") for p in GLCM_PARAM_NAMES)
        ]
    elif group == "glrlm":
        selected = hist_cols + [
            c for c in all_columns
            if any(c.startswith(p + "_") for p in GLRLM_PARAM_NAMES)
        ]
    else:
        print(f"[EROARE] Grup necunoscut: '{group}'. Valori acceptate: all, glcm, glrlm.")
        sys.exit(1)

    if not selected:
        print(f"[EROARE] Niciun parametru gasit pentru grupul '{group}' in CSV.")
        sys.exit(1)

    return selected


# ──────────────────────────── DETECTARE AUTOMATA PARAMETRI ──────────────────

def detect_feature_columns(df: pd.DataFrame) -> list:
    candidate_columns = [c for c in df.columns if c not in EXCLUDED_COLUMNS]
    feature_columns = []
    for col in candidate_columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().mean() >= 0.95:
            feature_columns.append(col)
    return feature_columns


# ──────────────────────────── INCARCARE DATE ────────────────────────────────

def load_data(csv_path: Path, group: str):
    if not csv_path.exists():
        print(f"[EROARE] Nu gasesc CSV-ul la: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    required_cols = {"probe", "section"}
    if not required_cols.issubset(df.columns):
        print(f"[EROARE] CSV-ul trebuie sa contina coloanele {required_cols}.")
        sys.exit(1)

    df["section_norm"] = df["section"].astype(str).str[0]
    df["probe_base"]   = df["probe"].astype(str).str.replace("HE", "", regex=False)
    df["key"]          = df["probe_base"] + "_" + df["section_norm"]
    df["diagnosis"]    = df["key"].map(DIAGNOSIS)

    df = df.dropna(subset=["diagnosis"])  # pastreaza toate cele 3 clase

    all_feature_columns = detect_feature_columns(df)
    feature_columns = filter_columns_by_group(all_feature_columns, group)

    for col in feature_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    n_before = len(df)
    df = df.dropna(subset=feature_columns)
    n_after = len(df)
    if n_after < n_before:
        print(f"  Atentie: {n_before - n_after} randuri eliminate (valori lipsa)")

    print(f"\n  CSV incarcat: {csv_path.name}")
    print(f"  Total imagini: {len(df)}")
    for c in CLASSES:
        n_img   = (df["diagnosis"] == c).sum()
        n_group = df.loc[df["diagnosis"] == c, "key"].nunique()
        print(f"  {c.capitalize():<10}: {n_img:>4} imagini | {n_group} grupuri de tesut")
    print(f"  Parametri detectati ({group}): {len(feature_columns)}")

    return df, feature_columns


# ──────────────────────────── CROSS-VALIDATION ──────────────────────────────

def cross_validate(df, feature_columns, n_folds, n_repeats):
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
    print(f"  Repetari       : {n_repeats}")
    print(f"  Fold-uri       : {n_folds}\n")

    acc_results, bal_acc_results = [], []
    all_y_true, all_y_pred = [], []

    for repeat in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                               random_state=RANDOM_STATE + repeat)

        for train_idx, test_idx in skf.split(keys, classes):
            train_keys = keys[train_idx]
            test_keys  = keys[test_idx]

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

            model = SVC(kernel=KERNEL, class_weight="balanced", random_state=RANDOM_STATE)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc_results.append(accuracy_score(y_test, y_pred))
            bal_acc_results.append(balanced_accuracy_score(y_test, y_pred))
            all_y_true.extend(y_test.tolist())
            all_y_pred.extend(y_pred.tolist())

        if (repeat + 1) % 5 == 0:
            print(f"  Repetarea {repeat+1:2d}/{n_repeats} "
                  f"— accuracy: {np.mean(acc_results)*100:.1f}% "
                  f"| balanced accuracy: {np.mean(bal_acc_results)*100:.1f}%")

    return acc_results, bal_acc_results, all_y_true, all_y_pred


# ──────────────────────────── VIZUALIZARI ───────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, csv_name, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    cm   = confusion_matrix(y_true, y_pred, labels=CLASSES)
    disp = ConfusionMatrixDisplay(cm, display_labels=[c.capitalize() for c in CLASSES])
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — SVM {KERNEL} 3-clase — {csv_name}", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / f"confusion_matrix_3class_{csv_name}.png", dpi=150)
    print(f"  Confusion matrix salvata")


def plot_accuracy_comparison(acc_results, bal_acc_results, csv_name, n_features, output_dir, n_folds, n_repeats):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    means = [np.mean(acc_results) * 100, np.mean(bal_acc_results) * 100]
    stds  = [np.std(acc_results) * 100, np.std(bal_acc_results) * 100]
    labels = ["Accuracy\n(bruta)", "Balanced Accuracy\n(medie per clasa)"]

    bars = ax.bar(labels, means, yerr=stds, capsize=6, color=["steelblue", "darkorange"])
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=13, fontweight="bold")

    ax.axhline(100 / 3, color="red", linestyle="--", linewidth=1.2, label="Random (33.3%)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"SVM rbf 3-clase — {csv_name} ({n_features} parametri)\n"
                 f"{n_repeats}x{n_folds}-Fold CV")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"accuracy_vs_balanced_3class_{csv_name}.png", dpi=150)
    print(f"  Grafic accuracy vs balanced accuracy salvat")


def plot_feature_importance(df, feature_columns, csv_name, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    X = df[feature_columns].astype(float)
    y = df["diagnosis"]

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = SVC(kernel=KERNEL, class_weight="balanced", random_state=RANDOM_STATE)
    model.fit(X_scaled, y)

    print(f"\n  Calculare Permutation Importance (3-clase, balanced_accuracy)...")
    perm = permutation_importance(
        model, X_scaled, y, n_repeats=10,
        random_state=RANDOM_STATE, scoring="balanced_accuracy"
    )

    indices = np.argsort(perm.importances_mean)[::-1]
    top_n   = min(20, len(feature_columns))
    top_idx = indices[:top_n]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(top_n), perm.importances_mean[top_idx],
           yerr=perm.importances_std[top_idx], color="steelblue", capsize=3)
    ax.set_xticks(range(top_n))
    ax.set_xticklabels([feature_columns[i] for i in top_idx], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Scadere medie balanced accuracy")
    ax.set_title(f"Permutation Importance — SVM {KERNEL} 3-clase — {csv_name} (Top 20)", fontsize=12)
    ax.axhline(0, color="red", linestyle="--", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(output_dir / f"feature_importance_3class_{csv_name}.png", dpi=150)
    print(f"  Feature importance salvata")

    print(f"\n  Top 10 parametri (dupa balanced accuracy):")
    print(f"  {'Parametru':<35} {'Importanta':>12} {'Std':>8}")
    print(f"  {'-'*57}")
    for i in indices[:10]:
        print(f"  {feature_columns[i]:<35} {perm.importances_mean[i]:>12.4f} {perm.importances_std[i]:>8.4f}")


# ──────────────────────────── ENTRY POINT ───────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Clasificare SVM generica 3-clase (normal/malign/displazic)"
    )
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--group", type=str, default="all", choices=["all", "glcm", "glrlm"])
    parser.add_argument("--n-repeats", type=int, default=25)
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    csv_name   = args.csv.stem
    group_tag  = f"_{args.group}" if args.group != "all" else ""
    csv_label  = f"{csv_name}{group_tag}"
    output_dir = args.output_dir or (args.csv.parent.parent / "Plots" / f"svm_3class_{csv_label}")

    print(f"\n{'='*60}")
    print(f"  Clasificare SVM 3-clase (normal/malign/displazic)")
    print(f"  CSV: {args.csv}")
    print(f"  Grup: {args.group} | {args.n_repeats}x{args.n_folds}-Fold CV")
    print(f"{'='*60}")

    df, feature_columns = load_data(args.csv, args.group)

    acc_results, bal_acc_results, all_y_true, all_y_pred = cross_validate(
        df, feature_columns, args.n_folds, args.n_repeats
    )

    mean_acc     = np.mean(acc_results)
    mean_bal_acc = np.mean(bal_acc_results)

    print(f"\n{'='*60}")
    print(f"  REZULTATE FINALE — {csv_label} ({len(feature_columns)} parametri)")
    print(f"{'='*60}")
    print(f"  Accuracy medie (bruta)  : {mean_acc*100:.1f}% (+/- {np.std(acc_results)*100:.1f}%)")
    print(f"  Balanced accuracy medie : {mean_bal_acc*100:.1f}% (+/- {np.std(bal_acc_results)*100:.1f}%)")

    print(f"\n  Raport detaliat:")
    print(classification_report(all_y_true, all_y_pred, labels=CLASSES, zero_division=0))

    print(f"\n  Generare grafice...")
    plot_confusion_matrix(all_y_true, all_y_pred, csv_label, output_dir)
    plot_accuracy_comparison(acc_results, bal_acc_results, csv_label,
                             len(feature_columns), output_dir, args.n_folds, args.n_repeats)
    plot_feature_importance(df, feature_columns, csv_label, output_dir)

    plt.close("all")

    print(f"\n{'='*60}")
    print(f"  Gata! Graficele sunt in: {output_dir}")
    print(f"{'='*60}\n")