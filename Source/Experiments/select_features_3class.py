"""
select_features_3class.py
Selectie de parametri (Lasso L1 sau RFE) combinata cu clasificare SVM
in 3 clase (normal/malign/displazic), generica pentru orice CSV al
proiectului -- extensie multiclasa a select_features.py / select_features_rfe.py.

Diferenta fata de select_features.py (binar):
    - LogisticRegression cu solver='liblinear' foloseste automat schema
      One-vs-Rest pentru multiclasa, deci coef_ devine o matrice
      (n_clase x n_parametri) in loc de un singur vector.
    - Un parametru e considerat "selectat" (Lasso) daca are coeficient
      nenul pentru CEL PUTIN UNA dintre cele 3 clase.
    - RFE gestioneaza automat coef_ multiclasa (norma pe toate clasele).
    - Raporteaza atat accuracy bruta cat si balanced accuracy (esentiala
      din cauza dezechilibrului 9 normal / 9 malign / 5 displazic).

Utilizare:
    python select_features_3class.py --csv "...\\features_glrlm.csv" --method lasso
    python select_features_3class.py --csv "...\\features_glrlm.csv" --method rfe --n-features 25
    python select_features_3class.py --csv "...\\features.csv" --method lasso --lasso-c 0.5
"""

import argparse
import sys
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report
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

EXCLUDED_COLUMNS = {
    "probe", "section", "polar", "channel", "is_HE", "filepath",
    "patch_row", "patch_col",
    "section_norm", "probe_base", "key", "diagnosis", "group_id",
    "dominant_class", "polar_num",
}

GLCM_PARAM_NAMES  = ["ASM", "Contrast", "Correlation", "IDM", "Entropy"]
GLRLM_PARAM_NAMES = ["SRE", "LRE", "GLN", "RLN", "RP"]
HISTOGRAM_NAMES   = ["Mean", "StDev", "Skew", "Kurt"]


def filter_columns_by_group(all_columns, group):
    if group == "all":
        return all_columns
    hist_cols = [c for c in all_columns if c in HISTOGRAM_NAMES]
    if group == "glcm":
        selected = hist_cols + [c for c in all_columns
                                 if any(c.startswith(p + "_") for p in GLCM_PARAM_NAMES)]
    elif group == "glrlm":
        selected = hist_cols + [c for c in all_columns
                                 if any(c.startswith(p + "_") for p in GLRLM_PARAM_NAMES)]
    else:
        print(f"[EROARE] Grup necunoscut: '{group}'.")
        sys.exit(1)
    if not selected:
        print(f"[EROARE] Niciun parametru gasit pentru grupul '{group}'.")
        sys.exit(1)
    return selected


def detect_feature_columns(df):
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
    if len(df) < n_before:
        print(f"  Atentie: {n_before - len(df)} randuri eliminate (valori lipsa)")

    print(f"\n  CSV incarcat: {csv_path.name}")
    for c in CLASSES:
        n_img   = (df["diagnosis"] == c).sum()
        n_group = df.loc[df["diagnosis"] == c, "key"].nunique()
        print(f"  {c.capitalize():<10}: {n_img:>4} imagini | {n_group} grupuri de tesut")
    print(f"  Parametri detectati ({group}): {len(feature_columns)}")

    return df, feature_columns


# ──────────────────────────── SELECTIE LASSO / RFE ───────────────────────────

def _get_ovr_coefs(ovr_classifier):
    """Extrage coeficientii din fiecare clasificator binar intern al unui
    OneVsRestClassifier deja antrenat, ca matrice (n_clase, n_parametri).
    Necesar pentru ca .coef_ nu mai e expus direct pe OneVsRestClassifier
    in versiunile noi de sklearn."""
    return np.vstack([est.coef_.ravel() for est in ovr_classifier.estimators_])


def select_features_lasso(X_train_scaled, y_train, feature_names, lasso_c):
    """LogisticRegression L1 impachetat explicit in OneVsRestClassifier
    (versiunile noi de sklearn nu mai fac OvR automat pentru liblinear
    la multiclasa). Parametru selectat daca are coeficient nenul pentru
    cel putin o clasa."""
    base = LogisticRegression(
        penalty="l1", solver="liblinear", C=lasso_c,
        class_weight="balanced", random_state=RANDOM_STATE, max_iter=5000,
    )
    lasso = OneVsRestClassifier(base)
    lasso.fit(X_train_scaled, y_train)

    coefs = _get_ovr_coefs(lasso)  # shape (n_classes, n_features)
    nonzero_any_class = np.any(coefs != 0, axis=0)
    selected = [feature_names[i] for i in range(len(feature_names)) if nonzero_any_class[i]]

    if len(selected) == 0:
        best_idx = np.argmax(np.abs(coefs).sum(axis=0))
        selected = [feature_names[best_idx]]

    return selected


def _ovr_importance_getter(ovr_classifier):
    """Importance_getter pentru RFE: norma L1 a coeficientilor peste
    toate cele 3 clase (OvR), per parametru."""
    coefs = _get_ovr_coefs(ovr_classifier)
    return np.abs(coefs).sum(axis=0)


def select_features_rfe(X_train_scaled, y_train, feature_names, n_features, rfe_step):
    if n_features >= len(feature_names):
        return feature_names[:]
    base = LogisticRegression(
        solver="liblinear", class_weight="balanced",
        random_state=RANDOM_STATE, max_iter=5000,
    )
    estimator = OneVsRestClassifier(base)
    rfe = RFE(
        estimator=estimator,
        n_features_to_select=n_features,
        step=rfe_step,
        importance_getter=_ovr_importance_getter,
    )
    rfe.fit(X_train_scaled, y_train)
    return [feature_names[i] for i in range(len(feature_names)) if rfe.support_[i]]


# ──────────────────────────── CROSS-VALIDATION ──────────────────────────────

def cross_validate(df, feature_columns, use_selection, method, lasso_c, n_features,
                    rfe_step, n_folds, n_repeats, label):
    probe_sections = (
        df.groupby("key")["diagnosis"]
        .agg(lambda x: x.value_counts().index[0])
        .reset_index().rename(columns={"diagnosis": "dominant_class"})
    )
    keys    = probe_sections["key"].values
    classes = probe_sections["dominant_class"].values

    acc_results, bal_acc_results = [], []
    all_y_true, all_y_pred = [], []
    selected_logs = []

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

            X_train_full = train_df[feature_columns].astype(float)
            y_train      = train_df["diagnosis"]
            X_test_full  = test_df[feature_columns].astype(float)
            y_test       = test_df["diagnosis"]

            scaler = StandardScaler()
            X_train_full_scaled = scaler.fit_transform(X_train_full)
            X_test_full_scaled  = scaler.transform(X_test_full)

            if use_selection:
                if method == "lasso":
                    selected = select_features_lasso(X_train_full_scaled, y_train,
                                                       feature_columns, lasso_c)
                else:
                    selected = select_features_rfe(X_train_full_scaled, y_train,
                                                     feature_columns, n_features, rfe_step)
                selected_logs.append(selected)
                idx_sel = [feature_columns.index(f) for f in selected]
                X_train = X_train_full_scaled[:, idx_sel]
                X_test  = X_test_full_scaled[:, idx_sel]
            else:
                X_train = X_train_full_scaled
                X_test  = X_test_full_scaled

            model = SVC(kernel=KERNEL, class_weight="balanced", random_state=RANDOM_STATE)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc_results.append(accuracy_score(y_test, y_pred))
            bal_acc_results.append(balanced_accuracy_score(y_test, y_pred))
            all_y_true.extend(y_test.tolist())
            all_y_pred.extend(y_pred.tolist())

        if (repeat + 1) % 5 == 0:
            print(f"  [{label}] Repetarea {repeat+1:2d}/{n_repeats} "
                  f"— acc: {np.mean(acc_results)*100:.1f}% "
                  f"| bal.acc: {np.mean(bal_acc_results)*100:.1f}%")

    return acc_results, bal_acc_results, all_y_true, all_y_pred, selected_logs


# ──────────────────────────── ANALIZA SELECTIE ──────────────────────────────

def analyze_selection(selected_logs, feature_columns):
    counter = Counter()
    sizes = []
    for selected in selected_logs:
        sizes.append(len(selected))
        counter.update(selected)
    total_folds = len(selected_logs)

    print(f"\n  Numar mediu de parametri selectati per fold: "
          f"{np.mean(sizes):.1f} din {len(feature_columns)}")

    top_n = min(15, len(feature_columns))
    print(f"\n  Top {top_n} parametri cei mai frecvent selectati "
          f"(din {total_folds} fold-uri):")
    print(f"  {'Parametru':<35} {'Frecventa':>12}")
    print(f"  {'-'*48}")
    for feat, count in counter.most_common(top_n):
        pct = count / total_folds * 100
        print(f"  {feat:<35} {count:>4}/{total_folds} ({pct:>5.1f}%)")

    return counter, sizes


# ──────────────────────────── VIZUALIZARI ───────────────────────────────────

def plot_comparison(acc_full, bal_full, acc_sel, bal_sel, csv_name, method_label,
                     output_dir, n_folds, n_repeats):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    labels = ["Toti\nparametrii\n(acc)", "Toti\nparametrii\n(bal.acc)",
              f"{method_label}\n(acc)", f"{method_label}\n(bal.acc)"]
    means  = [np.mean(acc_full)*100, np.mean(bal_full)*100,
              np.mean(acc_sel)*100, np.mean(bal_sel)*100]
    stds   = [np.std(acc_full)*100, np.std(bal_full)*100,
              np.std(acc_sel)*100, np.std(bal_sel)*100]
    colors = ["steelblue", "steelblue", "darkorange", "darkorange"]

    bars = ax.bar(labels, means, yerr=stds, capsize=5, color=colors)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.axhline(100/3, color="red", linestyle="--", linewidth=1, label="Random (33.3%)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"SVM rbf 3-clase — {csv_name}\ncu vs fara selectie {method_label} "
                 f"({n_repeats}x{n_folds}-Fold CV)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out_path = output_dir / f"{method_label.lower()}_comparison_3class_{csv_name}.png"
    plt.savefig(out_path, dpi=150)
    print(f"\n  Grafic comparatie salvat in {out_path}")


def plot_selection_frequency(counter, total_folds, csv_name, method_label, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    top_n = min(20, len(counter))
    items = counter.most_common(top_n)
    names = [x[0] for x in items]
    freqs = [x[1] / total_folds * 100 for x in items]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(len(names)), freqs, color="darkorange")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Frecventa selectie (%)")
    ax.set_title(f"Cei mai frecvent selectati parametri — {method_label} 3-clase — "
                 f"{csv_name} (Top {top_n})")
    ax.axhline(50, color="red", linestyle="--", linewidth=0.8,
               label="Selectat in >50% din fold-uri")
    ax.legend()
    plt.tight_layout()
    out_path = output_dir / f"{method_label.lower()}_frequency_3class_{csv_name}.png"
    plt.savefig(out_path, dpi=150)
    print(f"  Grafic frecventa selectie salvat in {out_path}")


# ──────────────────────────── ENTRY POINT ───────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Selectie de parametri (Lasso/RFE) + clasificare SVM 3-clase"
    )
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--method", type=str, default="lasso", choices=["lasso", "rfe"])
    parser.add_argument("--group", type=str, default="all", choices=["all", "glcm", "glrlm"])
    parser.add_argument("--lasso-c", type=float, default=0.5)
    parser.add_argument("--n-features", type=int, default=25,
                         help="Numar fix de parametri pentru RFE (implicit 25)")
    parser.add_argument("--rfe-step", type=int, default=3)
    parser.add_argument("--n-repeats", type=int, default=25)
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    csv_name    = args.csv.stem
    group_tag   = f"_{args.group}" if args.group != "all" else ""
    csv_label   = f"{csv_name}{group_tag}"
    method_label = "Lasso" if args.method == "lasso" else "RFE"
    output_dir  = args.output_dir or (args.csv.parent.parent / "Plots" /
                                        f"{args.method}_3class_{csv_label}")

    print(f"\n{'='*60}")
    print(f"  Selectie {method_label} + SVM 3-clase — {csv_label}")
    print(f"{'='*60}")

    df, feature_columns = load_data(args.csv, args.group)

    print(f"\n{'='*60}")
    print(f"  A) SVM rbf — toti cei {len(feature_columns)} parametri")
    print(f"{'='*60}")
    acc_full, bal_full, yt_full, yp_full, _ = cross_validate(
        df, feature_columns, use_selection=False, method=args.method,
        lasso_c=args.lasso_c, n_features=args.n_features, rfe_step=args.rfe_step,
        n_folds=args.n_folds, n_repeats=args.n_repeats, label="Toti"
    )

    print(f"\n{'='*60}")
    print(f"  B) SVM rbf — parametri selectati de {method_label}")
    print(f"{'='*60}")
    acc_sel, bal_sel, yt_sel, yp_sel, selected_logs = cross_validate(
        df, feature_columns, use_selection=True, method=args.method,
        lasso_c=args.lasso_c, n_features=args.n_features, rfe_step=args.rfe_step,
        n_folds=args.n_folds, n_repeats=args.n_repeats, label=method_label
    )

    print(f"\n{'='*60}")
    print(f"  Analiza parametrilor selectati")
    print(f"{'='*60}")
    counter, sizes = analyze_selection(selected_logs, feature_columns)

    print(f"\n{'='*60}")
    print(f"  REZULTATE FINALE — {csv_label}")
    print(f"{'='*60}")
    print(f"  {'Varianta':<30} {'Accuracy':>10} {'Bal.Acc':>10}")
    print(f"  {'-'*52}")
    print(f"  {'Toti (' + str(len(feature_columns)) + ')':<30} "
          f"{np.mean(acc_full)*100:>9.1f}% {np.mean(bal_full)*100:>9.1f}%")
    print(f"  {method_label + ' (~' + str(round(np.mean(sizes))) + ')':<30} "
          f"{np.mean(acc_sel)*100:>9.1f}% {np.mean(bal_sel)*100:>9.1f}%")
    print(f"\n  Diferenta balanced accuracy (Toti - {method_label}): "
          f"{(np.mean(bal_full) - np.mean(bal_sel))*100:+.1f} puncte procentuale")

    print(f"\n  Raport clasificare — toti parametrii:")
    print(classification_report(yt_full, yp_full, labels=CLASSES, zero_division=0))
    print(f"\n  Raport clasificare — {method_label}:")
    print(classification_report(yt_sel, yp_sel, labels=CLASSES, zero_division=0))

    print(f"\n  Generare grafice...")
    plot_comparison(acc_full, bal_full, acc_sel, bal_sel, csv_label, method_label,
                     output_dir, args.n_folds, args.n_repeats)
    plot_selection_frequency(counter, len(selected_logs), csv_label, method_label, output_dir)

    plt.close("all")

    print(f"\n{'='*60}")
    print(f"  Gata! Graficele sunt in: {output_dir}")
    print(f"{'='*60}\n")