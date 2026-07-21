"""
classify_svm_3class_patches.py
Clasificare SVM 3-clase (normal/malign/displazic) folosind PATCH-URILE
256x256 (Results/CSV/features_patches_256_corners.csv) in loc de
imaginile 512x512 intregi -- ideea fiind sa oferim SVM-ului mai multe
exemple de antrenare per grup de tesut (pana la 4x, cate un patch per
colt), ceea ce ar putea ajuta mai ales clasa minoritara (displazic,
doar 5 grupuri de tesut).

Metodologie:
    1. Antrenare: fiecare patch e un exemplu independent de antrenare
       (SVM invata pe toate patch-urile grupurilor din fold-ul de train).
    2. Evaluare DUBLA:
       a) PATCH-LEVEL: fiecare patch de test e evaluat individual
          (util pentru a vedea daca patch-urile in sine sunt informative).
       b) SECTION-LEVEL (agregat): predictiile tuturor patch-urilor unei
          sectiuni de test sunt combinate prin VOT MAJORITAR intr-o
          singura decizie per sectiune -- aceasta e metrica principala,
          comparabila direct cu rezultatele pe imagini intregi
          (Sectiunea 11.7).

Grupare CV identica cu restul proiectului: la nivel de grup de tesut
(probe+sectiune), pentru a evita orice leakage intre patch-uri din
aceeasi imagine sau sectiuni pereche (1+11, 2+22, 3+33).

Utilizare:
    python classify_svm_3class_patches.py --csv "...\\features_patches_256_corners.csv"
"""

import argparse
import sys
from collections import Counter
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
    n_before = len(df)
    df = df.dropna(subset=feature_columns)
    if len(df) < n_before:
        print(f"  Atentie: {n_before - len(df)} randuri eliminate (valori lipsa)")

    print(f"\n  CSV incarcat: {csv_path.name}")
    print(f"  Total patch-uri: {len(df)}")
    for c in CLASSES:
        n_patch = (df["diagnosis"] == c).sum()
        n_group = df.loc[df["diagnosis"] == c, "key"].nunique()
        print(f"  {c.capitalize():<10}: {n_patch:>5} patch-uri | {n_group} grupuri de tesut "
              f"(~{n_patch / n_group:.1f} patch-uri/grup)")
    print(f"  Parametri detectati: {len(feature_columns)}")

    return df, feature_columns


# ──────────────────────────── CROSS-VALIDATION CU AGREGARE ──────────────────

def cross_validate(df, feature_columns, n_folds, n_repeats):
    probe_sections = (
        df.groupby("key")["diagnosis"]
        .agg(lambda x: x.value_counts().index[0])
        .reset_index().rename(columns={"diagnosis": "dominant_class"})
    )
    keys    = probe_sections["key"].values
    classes = probe_sections["dominant_class"].values

    print(f"\n  Sectiuni unice : {len(keys)}")
    for c in CLASSES:
        print(f"  {c.capitalize():<10}: {(classes == c).sum()}")
    print(f"  Repetari       : {n_repeats}")
    print(f"  Fold-uri       : {n_folds}\n")

    # Rezultate PATCH-LEVEL
    acc_patch, bal_acc_patch = [], []
    y_true_patch, y_pred_patch = [], []

    # Rezultate SECTION-LEVEL (agregat prin vot majoritar -- hard)
    acc_section, bal_acc_section = [], []
    y_true_section, y_pred_section = [], []

    # Rezultate SECTION-LEVEL (agregat prin vot soft -- medie probabilitati)
    acc_section_soft, bal_acc_section_soft = [], []
    y_true_section_soft, y_pred_section_soft = [], []

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

            # probability=True e necesar pentru predict_proba (vot soft).
            # Foloseste Platt scaling intern (CV suplimentar pe train) --
            # antrenarea devine mai lenta, dar necesara pentru soft voting.
            model = SVC(kernel=KERNEL, class_weight="balanced",
                        random_state=RANDOM_STATE, probability=True)
            model.fit(X_train, y_train)
            y_pred  = model.predict(X_test)
            y_proba = model.predict_proba(X_test)  # shape (n_patch_test, 3)
            model_classes = model.classes_  # ordinea claselor in y_proba

            # --- PATCH-LEVEL ---
            acc_patch.append(accuracy_score(y_test, y_pred))
            bal_acc_patch.append(balanced_accuracy_score(y_test, y_pred))
            y_true_patch.extend(y_test.tolist())
            y_pred_patch.extend(y_pred.tolist())

            # --- SECTION-LEVEL: vot MAJORITAR (hard) pe eticheta finala ---
            test_df_copy = test_df.copy()
            test_df_copy["y_pred"] = y_pred
            for i, c in enumerate(model_classes):
                test_df_copy[f"proba_{c}"] = y_proba[:, i]

            section_true, section_pred_hard, section_pred_soft = [], [], []
            for key in test_keys:
                section_patches = test_df_copy[test_df_copy["key"] == key]
                if len(section_patches) == 0:
                    continue
                true_label = section_patches["diagnosis"].iloc[0]

                # Hard voting: eticheta cea mai frecventa
                pred_hard = Counter(section_patches["y_pred"]).most_common(1)[0][0]

                # Soft voting: media probabilitatilor pe toate patch-urile
                # sectiunii, apoi argmax
                mean_probas = section_patches[[f"proba_{c}" for c in model_classes]].mean(axis=0)
                pred_soft = model_classes[np.argmax(mean_probas.values)]

                section_true.append(true_label)
                section_pred_hard.append(pred_hard)
                section_pred_soft.append(pred_soft)

            if len(section_true) > 0:
                acc_section.append(accuracy_score(section_true, section_pred_hard))
                bal_acc_section.append(balanced_accuracy_score(section_true, section_pred_hard))
                y_true_section.extend(section_true)
                y_pred_section.extend(section_pred_hard)

                acc_section_soft.append(accuracy_score(section_true, section_pred_soft))
                bal_acc_section_soft.append(balanced_accuracy_score(section_true, section_pred_soft))
                y_true_section_soft.extend(section_true)
                y_pred_section_soft.extend(section_pred_soft)

        if (repeat + 1) % 5 == 0:
            print(f"  Repetarea {repeat+1:2d}/{n_repeats} "
                  f"— patch: {np.mean(bal_acc_patch)*100:.1f}% "
                  f"| hard: {np.mean(bal_acc_section)*100:.1f}% "
                  f"| soft: {np.mean(bal_acc_section_soft)*100:.1f}%")

    return {
        "patch": (acc_patch, bal_acc_patch, y_true_patch, y_pred_patch),
        "section": (acc_section, bal_acc_section, y_true_section, y_pred_section),
        "section_soft": (acc_section_soft, bal_acc_section_soft,
                          y_true_section_soft, y_pred_section_soft),
    }


# ──────────────────────────── VIZUALIZARI ───────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, title_suffix, filename, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    cm   = confusion_matrix(y_true, y_pred, labels=CLASSES)
    disp = ConfusionMatrixDisplay(cm, display_labels=[c.capitalize() for c in CLASSES])
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — SVM {KERNEL} 3-clase (patch-uri) — {title_suffix}",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=150)
    print(f"  Confusion matrix ({title_suffix}) salvata")


def plot_patch_vs_section(results, output_dir, n_folds, n_repeats):
    output_dir.mkdir(parents=True, exist_ok=True)
    acc_p, bal_p, _, _ = results["patch"]
    acc_s, bal_s, _, _ = results["section"]
    acc_ss, bal_ss, _, _ = results["section_soft"]

    fig, ax = plt.subplots(figsize=(11, 6))
    labels = ["Patch-level\n(acc)", "Patch-level\n(bal.acc)",
              "Sectiune HARD\n(acc)", "Sectiune HARD\n(bal.acc)",
              "Sectiune SOFT\n(acc)", "Sectiune SOFT\n(bal.acc)"]
    means = [np.mean(acc_p)*100, np.mean(bal_p)*100,
             np.mean(acc_s)*100, np.mean(bal_s)*100,
             np.mean(acc_ss)*100, np.mean(bal_ss)*100]
    stds  = [np.std(acc_p)*100, np.std(bal_p)*100,
             np.std(acc_s)*100, np.std(bal_s)*100,
             np.std(acc_ss)*100, np.std(bal_ss)*100]
    colors = ["lightsteelblue", "lightsteelblue", "darkorange", "darkorange",
              "seagreen", "seagreen"]

    bars = ax.bar(labels, means, yerr=stds, capsize=5, color=colors)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.axhline(100/3, color="red", linestyle="--", linewidth=1, label="Random (33.3%)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"SVM rbf 3-clase pe patch-uri — Patch-level vs Sectiune (Hard vs Soft voting)\n"
                 f"{n_repeats}x{n_folds}-Fold CV")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "patch_vs_section_3class.png", dpi=150)
    print(f"  Grafic patch vs sectiune (hard vs soft) salvat")


# ──────────────────────────── ENTRY POINT ───────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Clasificare SVM 3-clase pe patch-uri, cu agregare prin vot majoritar la nivel de sectiune"
    )
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--n-repeats", type=int, default=25)
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    csv_name   = args.csv.stem
    output_dir = args.output_dir or (args.csv.parent.parent / "Plots" / f"svm_3class_patches_{csv_name}")

    print(f"\n{'='*60}")
    print(f"  Clasificare SVM 3-clase pe PATCH-URI (cu agregare)")
    print(f"  CSV: {args.csv}")
    print(f"{'='*60}")

    df, feature_columns = load_data(args.csv)

    results = cross_validate(df, feature_columns, args.n_folds, args.n_repeats)

    acc_p, bal_p, yt_p, yp_p = results["patch"]
    acc_s, bal_s, yt_s, yp_s = results["section"]
    acc_ss, bal_ss, yt_ss, yp_ss = results["section_soft"]

    print(f"\n{'='*60}")
    print(f"  REZULTATE FINALE — {csv_name}")
    print(f"{'='*60}")
    print(f"  {'Nivel evaluare':<25} {'Accuracy':>10} {'Bal.Acc':>10}")
    print(f"  {'-'*47}")
    print(f"  {'Patch-level':<25} {np.mean(acc_p)*100:>9.1f}% {np.mean(bal_p)*100:>9.1f}%")
    print(f"  {'Sectiune HARD voting':<25} {np.mean(acc_s)*100:>9.1f}% {np.mean(bal_s)*100:>9.1f}%")
    print(f"  {'Sectiune SOFT voting':<25} {np.mean(acc_ss)*100:>9.1f}% {np.mean(bal_ss)*100:>9.1f}%")
    print(f"\n  Referinta (imagini 512x512, 64 parametri): 51.1% acc / 50.3% bal.acc")

    print(f"\n  Raport clasificare — PATCH-LEVEL:")
    print(classification_report(yt_p, yp_p, labels=CLASSES, zero_division=0))

    print(f"\n  Raport clasificare — SECTIUNE HARD voting:")
    print(classification_report(yt_s, yp_s, labels=CLASSES, zero_division=0))

    print(f"\n  Raport clasificare — SECTIUNE SOFT voting (metrica principala):")
    print(classification_report(yt_ss, yp_ss, labels=CLASSES, zero_division=0))

    print(f"\n  Generare grafice...")
    plot_confusion_matrix(yt_p, yp_p, "Patch-level",
                          f"confusion_matrix_patch_level_{csv_name}.png", output_dir)
    plot_confusion_matrix(yt_s, yp_s, "Sectiune HARD voting",
                          f"confusion_matrix_section_hard_{csv_name}.png", output_dir)
    plot_confusion_matrix(yt_ss, yp_ss, "Sectiune SOFT voting",
                          f"confusion_matrix_section_soft_{csv_name}.png", output_dir)
    plot_patch_vs_section(results, output_dir, args.n_folds, args.n_repeats)

    plt.close("all")

    print(f"\n{'='*60}")
    print(f"  Gata! Graficele sunt in: {output_dir}")
    print(f"{'='*60}\n")