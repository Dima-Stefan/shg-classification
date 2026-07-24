"""
select_features_corr_filter.py
Filtru de redundanta bazat pe corelatie, ghidat de un ranking de
importanta -- replica metodologiei descrise de profesor (bazata pe
ReliefF + filtru de corelatie Pearson), cu ReliefF INLOCUIT de
permutation importance (metrica deja folosita consecvent in acest
proiect, Sectiunea 6.5, 9.x), la solicitarea profesorului.

Algoritm (aplicat STRICT in interiorul fiecarui fold de train, identic
cu Lasso/RFE din select_features.py / select_features_rfe.py):

    1. Standardizare (StandardScaler, fit doar pe train).
    2. Ranking de relevanta: se antreneaza un SVM "de sondaj" pe toti
       parametrii (doar pe train), si se calculeaza permutation
       importance pentru fiecare parametru (inlocuieste ReliefF).
    3. Parametrii sunt sortati descrescator dupa importanta.
    4. Filtru de corelatie GREEDY, in ordinea rankingului: se parcurg
       parametrii de la cel mai important la cel mai putin important;
       un parametru e PASTRAT doar daca coeficientul Pearson (in
       valoare absoluta) fata de TOTI parametrii deja pastrati este
       sub prag (implicit 0.9). Altfel e eliminat (varianta cu ranking
       mai mic dintr-o pereche corelata e intotdeauna cea eliminata,
       pentru ca parametrii cu ranking mai mare au fost deja acceptati
       inaintea sa in parcurgere).
    5. SVM-ul FINAL se antreneaza doar pe parametrii ramasi; aceiasi
       indici (aceleasi coloane) sunt aplicati pe fold-ul de test.

Matricea de corelatie Pearson e calculata DOAR pe datele de train ale
fold-ului curent, pentru a evita orice scurgere de informatie din test.

Utilizare:
    python select_features_corr_filter.py --csv "...\\features.csv"
    python select_features_corr_filter.py --csv "...\\features_glrlm.csv" --corr-threshold 0.85
"""

import argparse
import sys
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report

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

    df = df.dropna(subset=["diagnosis"])
    df = df[df["diagnosis"] != "displazic"]   # clasificare binara, ca in select_features.py

    all_feature_columns = detect_feature_columns(df)
    feature_columns = filter_columns_by_group(all_feature_columns, group)

    for col in feature_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    n_before = len(df)
    df = df.dropna(subset=feature_columns)
    if len(df) < n_before:
        print(f"  Atentie: {n_before - len(df)} randuri eliminate (valori lipsa)")

    print(f"\n  CSV incarcat: {csv_path.name}")
    print(f"  Total imagini: {len(df)} | "
          f"normal: {(df['diagnosis']=='normal').sum()} | "
          f"malign: {(df['diagnosis']=='malign').sum()}")
    print(f"  Parametri detectati: {len(feature_columns)}")

    return df, feature_columns


# ──────────────────────────── RANKING + FILTRU DE CORELATIE ─────────────────

def rank_by_permutation_importance(X_train_scaled, y_train, feature_names, n_repeats_perm=5):
    """
    Inlocuitor pentru ReliefF: antreneaza un SVM 'de sondaj' pe toti
    parametrii disponibili si calculeaza permutation importance.
    Returneaza lista indicilor parametrilor, sortata descrescator
    dupa importanta.
    """
    probe_model = SVC(kernel=KERNEL, class_weight="balanced", random_state=RANDOM_STATE)
    probe_model.fit(X_train_scaled, y_train)

    perm = permutation_importance(
        probe_model, X_train_scaled, y_train,
        n_repeats=n_repeats_perm, random_state=RANDOM_STATE, scoring="accuracy"
    )
    ranked_indices = np.argsort(perm.importances_mean)[::-1]
    return ranked_indices


def greedy_correlation_filter(X_train_scaled, ranked_indices, corr_threshold):
    """
    Parcurge parametrii in ordinea rankingului (de la cel mai important).
    Un parametru e pastrat doar daca |Pearson| < prag fata de TOTI
    parametrii deja pastrati. Matricea de corelatie e calculata doar
    pe X_train_scaled (fold-ul curent de train).

    Returneaza lista indicilor pastrati, in ordinea rankingului.
    """
    corr_matrix = np.corrcoef(X_train_scaled, rowvar=False)

    kept_indices = []
    for idx in ranked_indices:
        if not kept_indices:
            kept_indices.append(idx)
            continue
        correlations_with_kept = np.abs(corr_matrix[idx, kept_indices])
        if np.all(correlations_with_kept < corr_threshold):
            kept_indices.append(idx)
        # altfel: eliminat -- corelat cu un parametru deja pastrat,
        # care are ranking mai mare (a fost procesat inaintea sa)

    return kept_indices


# ──────────────────────────── CROSS-VALIDATION ──────────────────────────────

def cross_validate(df, feature_columns, use_filter, corr_threshold, n_folds, n_repeats, label):
    probe_sections = (
        df.groupby("key")["diagnosis"]
        .agg(lambda x: x.value_counts().index[0])
        .reset_index().rename(columns={"diagnosis": "dominant_class"})
    )
    keys    = probe_sections["key"].values
    classes = probe_sections["dominant_class"].values

    results       = []
    all_y_true    = []
    all_y_pred    = []
    selected_logs = []

    for repeat in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                               random_state=RANDOM_STATE + repeat)

        for train_idx, test_idx in skf.split(keys, classes):
            train_keys = keys[train_idx]
            test_keys  = keys[test_idx]

            train_df = df[df["key"].isin(train_keys)]
            test_df  = df[df["key"].isin(test_keys)]

            if train_df["diagnosis"].nunique() < 2 or len(test_df) == 0:
                continue

            X_train_full = train_df[feature_columns].astype(float)
            y_train      = train_df["diagnosis"]
            X_test_full  = test_df[feature_columns].astype(float)
            y_test       = test_df["diagnosis"]

            scaler_full         = StandardScaler()
            X_train_full_scaled = scaler_full.fit_transform(X_train_full)
            X_test_full_scaled  = scaler_full.transform(X_test_full)

            if use_filter:
                ranked_indices = rank_by_permutation_importance(
                    X_train_full_scaled, y_train, feature_columns
                )
                kept_indices = greedy_correlation_filter(
                    X_train_full_scaled, ranked_indices, corr_threshold
                )
                selected = [feature_columns[i] for i in kept_indices]
                selected_logs.append(selected)

                X_train = X_train_full_scaled[:, kept_indices]
                X_test  = X_test_full_scaled[:, kept_indices]
            else:
                X_train = X_train_full_scaled
                X_test  = X_test_full_scaled

            model = SVC(kernel=KERNEL, class_weight="balanced", random_state=RANDOM_STATE)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            results.append(accuracy_score(y_test, y_pred))
            all_y_true.extend(y_test.tolist())
            all_y_pred.extend(y_pred.tolist())

        if (repeat + 1) % 5 == 0:
            print(f"  [{label}] Repetarea {repeat+1:2d}/{n_repeats} "
                  f"— acc medie: {np.mean(results)*100:.1f}%")

    return results, all_y_true, all_y_pred, selected_logs


# ──────────────────────────── ANALIZA SELECTIE ──────────────────────────────

def analyze_selection(selected_logs, feature_columns):
    counter = Counter()
    sizes   = []
    for selected in selected_logs:
        sizes.append(len(selected))
        counter.update(selected)
    total_folds = len(selected_logs)

    print(f"\n  Numar mediu de parametri pastrati per fold: "
          f"{np.mean(sizes):.1f} din {len(feature_columns)}")
    print(f"  Min: {min(sizes)} | Max: {max(sizes)}")

    top_n = min(15, len(feature_columns))
    print(f"\n  Top {top_n} parametri cei mai frecvent pastrati "
          f"(din {total_folds} fold-uri):")
    print(f"  {'Parametru':<35} {'Frecventa':>12}")
    print(f"  {'-'*48}")
    for feat, count in counter.most_common(top_n):
        pct = count / total_folds * 100
        print(f"  {feat:<35} {count:>4}/{total_folds} ({pct:>5.1f}%)")

    return counter, sizes


# ──────────────────────────── VIZUALIZARI ───────────────────────────────────

def plot_comparison(results_full, results_filter, csv_name, corr_threshold, output_dir, n_folds, n_repeats):
    output_dir.mkdir(parents=True, exist_ok=True)
    means  = [np.mean(results_full) * 100, np.mean(results_filter) * 100]
    stds   = [np.std(results_full)  * 100, np.std(results_filter)  * 100]
    labels = ["SVM rbf\n(toti parametrii)", f"SVM rbf\n(filtru corelatie r>{corr_threshold})"]
    colors = ["steelblue", "mediumpurple"]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(labels, means, color=colors, width=0.5, yerr=stds, capsize=6)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=13, fontweight="bold")

    ax.axhline(50, color="red", linestyle="--", linewidth=1.2, label="Random (50%)")
    ax.set_ylabel("Accuracy medie (%)", fontsize=12)
    ax.set_title(
        f"SVM rbf cu vs fara filtru de corelatie (ranking permutation importance)\n"
        f"{csv_name} — {n_repeats}x{n_folds}-Fold CV",
        fontsize=12
    )
    ax.set_ylim(0, 100)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out_path = output_dir / f"corr_filter_comparison_{csv_name}.png"
    plt.savefig(out_path, dpi=150)
    print(f"\n  Grafic comparatie salvat in {out_path}")


def plot_selection_frequency(counter, total_folds, csv_name, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    top_n = min(20, len(counter))
    items = counter.most_common(top_n)
    names = [x[0] for x in items]
    freqs = [x[1] / total_folds * 100 for x in items]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(len(names)), freqs, color="mediumpurple")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Frecventa pastrare (%)")
    ax.set_title(f"Cei mai frecvent pastrati parametri (filtru corelatie) — {csv_name} (Top {top_n})",
                 fontsize=12)
    ax.axhline(50, color="red", linestyle="--", linewidth=0.8,
               label="Pastrat in >50% din fold-uri")
    ax.legend()
    plt.tight_layout()
    out_path = output_dir / f"corr_filter_frequency_{csv_name}.png"
    plt.savefig(out_path, dpi=150)
    print(f"  Grafic frecventa pastrare salvat in {out_path}")


# ──────────────────────────── ENTRY POINT ───────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Filtru de corelatie ghidat de permutation importance (inlocuitor ReliefF)"
    )
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--group", type=str, default="all", choices=["all", "glcm", "glrlm"])
    parser.add_argument("--corr-threshold", type=float, default=0.9,
                         help="Prag Pearson peste care doi parametri sunt considerati redundanti (implicit 0.9)")
    parser.add_argument("--n-repeats", type=int, default=25)
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    csv_name   = args.csv.stem
    group_tag  = f"_{args.group}" if args.group != "all" else ""
    csv_label  = f"{csv_name}{group_tag}"
    output_dir = args.output_dir or (args.csv.parent.parent / "Plots" / f"corr_filter_{csv_label}")

    print(f"\n{'='*60}")
    print(f"  Filtru de corelatie (ranking: permutation importance, inlocuitor ReliefF)")
    print(f"  CSV: {args.csv}")
    print(f"  Grup: {args.group} | Prag corelatie: {args.corr_threshold} | "
          f"{args.n_repeats}x{args.n_folds}-Fold CV")
    print(f"{'='*60}")

    df, feature_columns = load_data(args.csv, args.group)

    print(f"\n{'='*60}")
    print(f"  A) SVM rbf — toti cei {len(feature_columns)} parametri (referinta)")
    print(f"{'='*60}")
    results_full, yt_full, yp_full, _ = cross_validate(
        df, feature_columns, use_filter=False, corr_threshold=args.corr_threshold,
        n_folds=args.n_folds, n_repeats=args.n_repeats, label="Toti parametrii"
    )

    print(f"\n{'='*60}")
    print(f"  B) SVM rbf — dupa filtru de corelatie (r>{args.corr_threshold})")
    print(f"{'='*60}")
    results_filter, yt_filter, yp_filter, selected_logs = cross_validate(
        df, feature_columns, use_filter=True, corr_threshold=args.corr_threshold,
        n_folds=args.n_folds, n_repeats=args.n_repeats, label="Filtru corelatie"
    )

    print(f"\n{'='*60}")
    print(f"  Analiza parametrilor pastrati")
    print(f"{'='*60}")
    counter, sizes = analyze_selection(selected_logs, feature_columns)

    mean_full   = np.mean(results_full)   * 100
    mean_filter = np.mean(results_filter) * 100

    print(f"\n{'='*60}")
    print(f"  REZULTATE FINALE — {csv_label} (prag corelatie: {args.corr_threshold})")
    print(f"{'='*60}")
    print(f"  {'Varianta':<32} {'Acc medie':>10} {'Std':>8}")
    print(f"  {'-'*52}")
    print(f"  {'Toti parametrii ('+str(len(feature_columns))+')':<32} {mean_full:>9.1f}% {np.std(results_full)*100:>7.1f}%")
    print(f"  {'Filtru corelatie (~'+str(round(np.mean(sizes)))+')':<32} {mean_filter:>9.1f}% {np.std(results_filter)*100:>7.1f}%")
    print(f"\n  Diferenta (Toti - Filtru): {mean_full - mean_filter:+.1f} puncte procentuale")
    print(f"  Reducere dimensionalitate: {len(feature_columns)} -> "
          f"~{round(np.mean(sizes))} parametri "
          f"({(1 - np.mean(sizes)/len(feature_columns))*100:.0f}% reducere)")

    print(f"\n  Raport clasificare — toti parametrii:")
    print(classification_report(yt_full, yp_full, labels=["normal", "malign"], zero_division=0))
    print(f"\n  Raport clasificare — filtru corelatie:")
    print(classification_report(yt_filter, yp_filter, labels=["normal", "malign"], zero_division=0))

    print(f"\n  Generare grafice...")
    plot_comparison(results_full, results_filter, csv_label, args.corr_threshold,
                    output_dir, args.n_folds, args.n_repeats)
    plot_selection_frequency(counter, len(selected_logs), csv_label, output_dir)

    plt.close("all")

    print(f"\n{'='*60}")
    print(f"  Gata! Graficele sunt in: {output_dir}")
    print(f"{'='*60}\n")