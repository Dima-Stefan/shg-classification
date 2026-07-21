"""
classify_svm_hierarchical_3class.py
Clasificare ierarhica in 2 PASI pentru 3 clase (normal/malign/displazic),
combinata cu patch-uri 256x256 + hard voting (cea mai buna combinatie
gasita pana acum, Sectiunea 11.8, 54.7% balanced accuracy).

Motivatie: in loc sa ceara SVM-ului sa separe direct 3 clase cu
dezechilibru sever (9/9/5), problema e descompusa in 2 sub-probleme
binare, fiecare cu dezechilibru mai mic sau structura mai simpla:

    PAS 1: normal (9 grupuri) vs anormal = malign+displazic (14 grupuri)
    PAS 2: malign (9 grupuri) vs displazic (5 grupuri)
           -- aplicat DOAR sectiunilor clasificate "anormal" la Pasul 1

Fiecare pas foloseste patch-uri + hard voting (vot majoritar pe cele
4 patch-uri ale unei sectiuni), identic metodologic cu
classify_svm_3class_patches.py.

Predictia finala per sectiune:
    daca Pas 1 -> "normal": predictie finala = normal
    daca Pas 1 -> "anormal": predictie finala = rezultatul Pasului 2
                              (malign sau displazic)

Grupare CV la nivel de grup de tesut (probe+sectiune), identic cu
restul proiectului -- Pasul 2 se antreneaza DOAR pe grupurile anormale
din fold-ul curent de train (subset al train_keys), fara nicio
scurgere de informatie din fold-ul de test.

Utilizare:
    python classify_svm_hierarchical_3class.py --csv "...\\features_patches_256_corners.csv"
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
    "dominant_class", "polar_num", "step1_label",
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

def load_data(csv_path: Path, first_class: str):
    if not csv_path.exists():
        print(f"[EROARE] Nu gasesc CSV-ul la: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    df["section_norm"] = df["section"].astype(str).str[0]
    df["probe_base"]   = df["probe"].astype(str).str.replace("HE", "", regex=False)
    df["key"]          = df["probe_base"] + "_" + df["section_norm"]
    df["diagnosis"]    = df["key"].map(DIAGNOSIS)
    df = df.dropna(subset=["diagnosis"])

    # Pas 1: first_class vs restul (celelalte doua combinate)
    other_label = f"non_{first_class}"
    df["step1_label"] = df["diagnosis"].apply(
        lambda d: first_class if d == first_class else other_label
    )
    step2_classes = [c for c in CLASSES if c != first_class]

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
        print(f"  {c.capitalize():<10}: {n_patch:>5} patch-uri | {n_group} grupuri de tesut")
    print(f"  Parametri detectati: {len(feature_columns)}")
    print(f"\n  Pas 1 -- {first_class.capitalize()}: "
          f"{(df['step1_label']==first_class).sum()} patch-uri | "
          f"{other_label.capitalize()}: {(df['step1_label']==other_label).sum()} patch-uri")
    print(f"  Pas 2 -- {step2_classes[0].capitalize()} vs {step2_classes[1].capitalize()}")

    return df, feature_columns, first_class, other_label, step2_classes


# (votarea per sectiune e acum implementata inline in cross_validate,
#  separat pentru fiecare pas, pentru a suporta hard/soft configurabil)


# ──────────────────────────── CROSS-VALIDATION IERARHICA ────────────────────

def cross_validate(df, feature_columns, first_class, other_label, step2_classes,
                    step1_voting, step2_voting, n_folds, n_repeats):
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

    # Rezultate PAS 1 (normal vs anormal), la nivel de sectiune
    acc_step1, bal_acc_step1 = [], []

    # Rezultate PAS 2 (malign vs displazic), evaluat DOAR pe sectiunile
    # cu adevarat malign/displazic (diagnostic separat al pasului, nu
    # depinde de reusita Pasului 1)
    acc_step2, bal_acc_step2 = [], []

    # Rezultate FINALE (3-clase, dupa ambii pasi combinati)
    acc_final, bal_acc_final = [], []
    y_true_final, y_pred_final = [], []

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

            # ── PAS 1: normal vs anormal ─────────────────────────────────
            X_train1 = train_df[feature_columns].astype(float)
            y_train1 = train_df["step1_label"]
            X_test1  = test_df[feature_columns].astype(float)

            scaler1 = StandardScaler()
            X_train1_scaled = scaler1.fit_transform(X_train1)
            X_test1_scaled  = scaler1.transform(X_test1)

            model1 = SVC(kernel=KERNEL, class_weight="balanced", random_state=RANDOM_STATE,
                         probability=(step1_voting == "soft"))
            model1.fit(X_train1_scaled, y_train1)

            test_df = test_df.copy()
            test_df["pred_step1"] = model1.predict(X_test1_scaled)

            if step1_voting == "soft":
                proba1 = model1.predict_proba(X_test1_scaled)
                for i, c in enumerate(model1.classes_):
                    test_df[f"proba1_{c}"] = proba1[:, i]

            def vote_step1(section_patches):
                if step1_voting == "hard":
                    return Counter(section_patches["pred_step1"]).most_common(1)[0][0]
                mean_probas = section_patches[[f"proba1_{c}" for c in model1.classes_]].mean(axis=0)
                return model1.classes_[np.argmax(mean_probas.values)]

            step1_section_pred = {}
            for key in test_keys:
                section_patches = test_df[test_df["key"] == key]
                if len(section_patches) == 0:
                    continue
                step1_section_pred[key] = vote_step1(section_patches)

            # Evaluare Pas 1 la nivel de sectiune (referinta: step1_label adevarat)
            step1_true_map = (
                test_df.drop_duplicates("key").set_index("key")["step1_label"].to_dict()
            )
            s1_true = [step1_true_map[k] for k in step1_section_pred]
            s1_pred = [step1_section_pred[k] for k in step1_section_pred]
            if len(s1_true) > 0:
                acc_step1.append(accuracy_score(s1_true, s1_pred))
                bal_acc_step1.append(balanced_accuracy_score(s1_true, s1_pred))

            # ── PAS 2: step2_classes[0] vs step2_classes[1], antrenat doar pe train != first_class ──
            train_df_other = train_df[train_df["diagnosis"] != first_class]
            X_train2 = train_df_other[feature_columns].astype(float)
            y_train2 = train_df_other["diagnosis"]

            scaler2 = StandardScaler()
            X_train2_scaled = scaler2.fit_transform(X_train2)

            model2 = SVC(kernel=KERNEL, class_weight="balanced", random_state=RANDOM_STATE,
                         probability=(step2_voting == "soft"))
            model2.fit(X_train2_scaled, y_train2)

            # Aplicam Pasul 2 pe TOATE patch-urile de test (pentru diagnostic
            # separat), dar folosim rezultatul doar pentru sectiunile
            # clasificate "first_class"-negativ la Pasul 1 (in predictia finala)
            X_test2_scaled = scaler2.transform(test_df[feature_columns].astype(float))
            test_df["pred_step2"] = model2.predict(X_test2_scaled)

            if step2_voting == "soft":
                proba2 = model2.predict_proba(X_test2_scaled)
                for i, c in enumerate(model2.classes_):
                    test_df[f"proba2_{c}"] = proba2[:, i]

            def vote_step2(section_patches):
                if step2_voting == "hard":
                    return Counter(section_patches["pred_step2"]).most_common(1)[0][0]
                mean_probas = section_patches[[f"proba2_{c}" for c in model2.classes_]].mean(axis=0)
                return model2.classes_[np.argmax(mean_probas.values)]

            step2_section_pred = {}
            for key in test_keys:
                section_patches = test_df[test_df["key"] == key]
                if len(section_patches) == 0:
                    continue
                step2_section_pred[key] = vote_step2(section_patches)

            # Evaluare Pas 2 DOAR pe sectiunile care nu sunt first_class
            true_diag_map = test_df.drop_duplicates("key").set_index("key")["diagnosis"].to_dict()
            other_test_keys = [k for k in test_keys if true_diag_map.get(k) != first_class]
            if len(other_test_keys) > 0:
                s2_true = [true_diag_map[k] for k in other_test_keys]
                s2_pred = [step2_section_pred[k] for k in other_test_keys]
                acc_step2.append(accuracy_score(s2_true, s2_pred))
                bal_acc_step2.append(balanced_accuracy_score(s2_true, s2_pred))

            # ── PREDICTIA FINALA (combinare Pas 1 + Pas 2) ──────────────────
            final_true, final_pred = [], []
            for k in test_keys:
                if k not in step1_section_pred:
                    continue
                true_label = true_diag_map[k]
                if step1_section_pred[k] == first_class:
                    pred_label = first_class
                else:
                    pred_label = step2_section_pred[k]
                final_true.append(true_label)
                final_pred.append(pred_label)

            if len(final_true) > 0:
                acc_final.append(accuracy_score(final_true, final_pred))
                bal_acc_final.append(balanced_accuracy_score(final_true, final_pred))
                y_true_final.extend(final_true)
                y_pred_final.extend(final_pred)

        if (repeat + 1) % 5 == 0:
            print(f"  Repetarea {repeat+1:2d}/{n_repeats} "
                  f"— Pas1 bal.acc: {np.mean(bal_acc_step1)*100:.1f}% "
                  f"| Pas2 bal.acc: {np.mean(bal_acc_step2)*100:.1f}% "
                  f"| FINAL bal.acc: {np.mean(bal_acc_final)*100:.1f}%")

    return {
        "step1": (acc_step1, bal_acc_step1),
        "step2": (acc_step2, bal_acc_step2),
        "final": (acc_final, bal_acc_final, y_true_final, y_pred_final),
    }


# ──────────────────────────── VIZUALIZARI ───────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, output_dir, csv_name):
    output_dir.mkdir(parents=True, exist_ok=True)
    cm   = confusion_matrix(y_true, y_pred, labels=CLASSES)
    disp = ConfusionMatrixDisplay(cm, display_labels=[c.capitalize() for c in CLASSES])
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — Clasificare IERARHICA (2 pasi) — {csv_name}", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_dir / f"confusion_matrix_hierarchical_{csv_name}.png", dpi=150)
    print(f"  Confusion matrix salvata")


def plot_summary(results, output_dir, csv_name, first_class, other_label, step2_classes, n_folds, n_repeats):
    output_dir.mkdir(parents=True, exist_ok=True)
    acc_s1, bal_s1 = results["step1"]
    acc_s2, bal_s2 = results["step2"]
    acc_f, bal_f   = results["final"][0], results["final"][1]

    fig, ax = plt.subplots(figsize=(10, 6))
    labels = [f"Pas 1\n{first_class} vs {other_label}\n(bal.acc)",
              f"Pas 2\n{step2_classes[0]} vs {step2_classes[1]}\n(bal.acc)",
              "FINAL\n3-clase\n(bal.acc)"]
    means = [np.mean(bal_s1)*100, np.mean(bal_s2)*100, np.mean(bal_f)*100]
    stds  = [np.std(bal_s1)*100, np.std(bal_s2)*100, np.std(bal_f)*100]
    colors = ["cornflowerblue", "mediumpurple", "seagreen"]

    bars = ax.bar(labels, means, yerr=stds, capsize=6, color=colors)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.axhline(50, color="gray", linestyle=":", linewidth=1, label="Random binar (50%)")
    ax.axhline(100/3, color="red", linestyle="--", linewidth=1, label="Random 3-clase (33.3%)")
    ax.set_ylabel("Balanced Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"Clasificare IERARHICA (2 pasi) pe patch-uri — {csv_name}\n"
                 f"{n_repeats}x{n_folds}-Fold CV")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"hierarchical_summary_{csv_name}.png", dpi=150)
    print(f"  Grafic sumar salvat")


# ──────────────────────────── ENTRY POINT ───────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Clasificare ierarhica 2 pasi (normal vs anormal, apoi malign vs displazic)"
    )
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--first-class", type=str, default="normal",
                         choices=["normal", "malign", "displazic"],
                         help="Clasa separata la Pasul 1 (implicit: normal)")
    parser.add_argument("--step1-voting", type=str, default="hard", choices=["hard", "soft"])
    parser.add_argument("--step2-voting", type=str, default="hard", choices=["hard", "soft"])
    parser.add_argument("--n-repeats", type=int, default=25)
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    csv_name   = args.csv.stem
    voting_tag = f"s1{args.step1_voting}_s2{args.step2_voting}"
    output_dir = args.output_dir or (args.csv.parent.parent / "Plots" /
                                       f"hierarchical_3class_{args.first_class}first_{voting_tag}_{csv_name}")

    print(f"\n{'='*60}")
    print(f"  Clasificare IERARHICA 3-clase (2 pasi) pe patch-uri")
    print(f"  Pas 1: {args.first_class} vs restul ({args.step1_voting} voting)")
    print(f"  Pas 2: ({args.step2_voting} voting)")
    print(f"  CSV: {args.csv}")
    print(f"{'='*60}")

    df, feature_columns, first_class, other_label, step2_classes = load_data(args.csv, args.first_class)

    results = cross_validate(df, feature_columns, first_class, other_label, step2_classes,
                              args.step1_voting, args.step2_voting, args.n_folds, args.n_repeats)

    acc_s1, bal_s1 = results["step1"]
    acc_s2, bal_s2 = results["step2"]
    acc_f, bal_f, yt_f, yp_f = results["final"]

    print(f"\n{'='*60}")
    print(f"  REZULTATE FINALE — {csv_name} (Pas 1: {first_class} vs restul)")
    print(f"{'='*60}")
    print(f"  {'Etapa':<40} {'Accuracy':>10} {'Bal.Acc':>10}")
    print(f"  {'-'*62}")
    print(f"  {'Pas 1: ' + first_class + ' vs ' + other_label:<40} {np.mean(acc_s1)*100:>9.1f}% {np.mean(bal_s1)*100:>9.1f}%")
    print(f"  {'Pas 2: ' + step2_classes[0] + ' vs ' + step2_classes[1]:<40} {np.mean(acc_s2)*100:>9.1f}% {np.mean(bal_s2)*100:>9.1f}%")
    print(f"  {'FINAL (3-clase combinat)':<40} {np.mean(acc_f)*100:>9.1f}% {np.mean(bal_f)*100:>9.1f}%")
    print(f"\n  Referinte anterioare:")
    print(f"    Imagini intregi, 3-clase directe : 51.1% acc / 50.3% bal.acc")
    print(f"    Patch-uri + hard voting, directe  : 55.9% acc / 54.7% bal.acc")
    print(f"    Ierarhic, normal primul           : 52.1% bal.acc")

    print(f"\n  Raport clasificare FINAL (3-clase):")
    print(classification_report(yt_f, yp_f, labels=CLASSES, zero_division=0))

    print(f"\n  Generare grafice...")
    plot_confusion_matrix(yt_f, yp_f, output_dir, f"{args.first_class}first_{csv_name}")
    plot_summary(results, output_dir, csv_name, first_class, other_label, step2_classes,
                 args.n_folds, args.n_repeats)

    plt.close("all")

    print(f"\n{'='*60}")
    print(f"  Gata! Graficele sunt in: {output_dir}")
    print(f"{'='*60}\n")