"""
select_features_rfe.py
Script generic si reutilizabil de selectie de parametri prin RFE
(Recursive Feature Elimination cu LogisticRegression ca estimator),
aplicabil pe orice CSV generat de pipeline-ul proiectului
(features.csv, features_glrlm.csv, features_corr_std.csv etc.)
fara sa fie nevoie de un script de clasificare nou pentru fiecare
combinatie.

Spre deosebire de select_features.py (Lasso), RFE elimina iterativ
parametrul cel mai putin important, reantrenand de fiecare data, pana
raman exact N parametri specificati explicit prin --n-features.

Coloanele de parametri sunt detectate AUTOMAT — identic cu
select_features.py.

Metodologie identica cu classify_svm_kfold_rfe.py: selectia RFE
se aplica STRICT in interiorul fiecarui fold de CV, folosind doar
datele de training ale acelui fold, pentru a evita orice leakage
prin alegerea parametrilor.

Utilizare:
    python select_features_rfe.py --csv "C:\\...\\Results\\features.csv" --n-features 25
    python select_features_rfe.py --csv "C:\\...\\Results\\features_glrlm.csv" --n-features 30
    python select_features_rfe.py --csv "C:\\...\\Results\\features_glrlm.csv" --n-features 20 --rfe-step 2

Argumente:
    --csv         Calea catre CSV-ul de analizat (OBLIGATORIU)
    --n-features  Numarul fix de parametri de pastrat prin RFE (OBLIGATORIU)
    --rfe-step    Cati parametri elimina RFE la fiecare pas intern
                  (implicit 3; mai mic = mai precis, mai lent)
    --n-repeats   Numarul de repetari ale CV (implicit 25)
    --n-folds     Numarul de fold-uri (implicit 3)
    --output-dir  Folderul pentru grafice (implicit langa CSV-ul de input)
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
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report

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

KERNEL       = "rbf"
RANDOM_STATE = 42

# Coloane care NU sunt parametri, indiferent daca sunt numerice sau nu.
# Orice coloana numerica ce NU e in aceasta lista e tratata ca parametru
# de clasificare.
EXCLUDED_COLUMNS = {
    "probe", "section", "polar", "channel", "is_HE", "filepath",
    "patch_row", "patch_col",
    # coloane auxiliare adaugate intern in timpul procesarii:
    "section_norm", "probe_base", "key", "diagnosis", "group_id",
    "dominant_class", "polar_num",
}


# ──────────────────────────── DETECTARE AUTOMATA PARAMETRI ──────────────────

def detect_feature_columns(df: pd.DataFrame) -> list:
    """
    Detecteaza automat coloanele de parametri dintr-un DataFrame:
    toate coloanele numerice, exceptand cele din EXCLUDED_COLUMNS.

    Foloseste pd.to_numeric cu errors='coerce' pe fiecare coloana
    candidata, ca sa prinda si coloane salvate ca string in CSV
    (ex. "8.37e-04" scris ca text de smart_format din main.py) dar
    care reprezinta de fapt valori numerice.
    """
    candidate_columns = [c for c in df.columns if c not in EXCLUDED_COLUMNS]

    feature_columns = []
    for col in candidate_columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        # Daca dupa conversie cel putin 95% din valori sunt numerice
        # (nu NaN), consideram coloana drept parametru numeric valid.
        # Pragul de 95% tolereaza eventuale valori lipsa izolate, fara
        # sa accepte din greseala coloane text (unde aproape tot ar fi NaN).
        non_null_ratio = converted.notna().mean()
        if non_null_ratio >= 0.95:
            feature_columns.append(col)

    return feature_columns


# ──────────────────────────── INCARCARE DATE ────────────────────────────────

def load_data(csv_path: Path):
    if not csv_path.exists():
        print(f"[EROARE] Nu gasesc CSV-ul la: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    required_cols = {"probe", "section"}
    if not required_cols.issubset(df.columns):
        print(f"[EROARE] CSV-ul trebuie sa contina coloanele {required_cols}. "
              f"Coloane gasite: {list(df.columns)}")
        sys.exit(1)

    df["section_norm"] = df["section"].astype(str).str[0]
    df["probe_base"]   = df["probe"].astype(str).str.replace("HE", "", regex=False)
    df["key"]          = df["probe_base"] + "_" + df["section_norm"]
    df["diagnosis"]    = df["key"].map(DIAGNOSIS)

    df = df.dropna(subset=["diagnosis"])
    df = df[df["diagnosis"] != "displazic"]

    feature_columns = detect_feature_columns(df)

    # Convertim explicit coloanele de parametri la float, in caz ca
    # au fost salvate ca text (ex. notatie stiintifica din main.py)
    for col in feature_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Eliminam eventualele randuri cu NaN ramase in parametri, dupa
    # conversie (de obicei 0, dar facem verificarea explicit)
    n_before = len(df)
    df = df.dropna(subset=feature_columns)
    n_after = len(df)
    if n_after < n_before:
        print(f"  Atentie: {n_before - n_after} randuri eliminate "
              f"(valori lipsa dupa conversia numerica)")

    print(f"\n  CSV incarcat: {csv_path.name}")
    print(f"  Total imagini: {len(df)} | "
          f"normal: {(df['diagnosis']=='normal').sum()} | "
          f"malign: {(df['diagnosis']=='malign').sum()}")
    print(f"  Parametri detectati automat: {len(feature_columns)}")
    print(f"  Lista: {feature_columns[:5]}{'...' if len(feature_columns) > 5 else ''}")

    return df, feature_columns


# ──────────────────────────── SELECTIE RFE ──────────────────────────────────

def select_features_rfe(X_train_scaled, y_train, feature_names, n_features, rfe_step):
    """
    Antreneaza RFE cu LogisticRegression ca estimator pe datele de
    train (deja scalate) si returneaza lista celor n_features
    parametri pastrati.

    RFE elimina iterativ cate rfe_step parametri cu coeficientul
    absolut cel mai mic, reantrenand de fiecare data, pana raman
    exact n_features parametri.

    Daca n_features >= len(feature_names), nu se face nicio eliminare
    si se returneaza toti parametrii (fallback de siguranta).
    """
    if n_features >= len(feature_names):
        return feature_names[:]

    estimator = LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        max_iter=5000,
    )

    rfe = RFE(
        estimator=estimator,
        n_features_to_select=n_features,
        step=rfe_step,
    )
    rfe.fit(X_train_scaled, y_train)

    selected = [feature_names[i] for i in range(len(feature_names)) if rfe.support_[i]]
    return selected


# ──────────────────────────── CROSS-VALIDATION ──────────────────────────────

def cross_validate(df, feature_columns, use_rfe, n_features, rfe_step, n_folds, n_repeats, label):
    """
    Stratified n_folds-Fold CV repetat n_repeats ori la nivel de
    proba+sectiune. Identic metodologic cu classify_svm_kfold_rfe.py,
    generalizat pentru orice lista de feature_columns.
    """
    probe_sections = (
        df.groupby("key")["diagnosis"]
        .agg(lambda x: x.value_counts().index[0])
        .reset_index()
        .rename(columns={"diagnosis": "dominant_class"})
    )

    keys    = probe_sections["key"].values
    classes = probe_sections["dominant_class"].values

    results       = []
    all_y_true    = []
    all_y_pred    = []
    selected_logs = []

    for repeat in range(n_repeats):
        skf = StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=RANDOM_STATE + repeat
        )

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

            if use_rfe:
                selected = select_features_rfe(
                    X_train_full_scaled, y_train, feature_columns,
                    n_features, rfe_step
                )
                selected_logs.append(selected)

                idx_selected = [feature_columns.index(f) for f in selected]
                X_train = X_train_full_scaled[:, idx_selected]
                X_test  = X_test_full_scaled[:, idx_selected]
            else:
                X_train = X_train_full_scaled
                X_test  = X_test_full_scaled

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
            print(f"  [{label}] Repetarea {repeat+1:2d}/{n_repeats} "
                  f"— acc medie: {np.mean(results)*100:.1f}%")

    return results, all_y_true, all_y_pred, selected_logs


# ──────────────────────────── ANALIZA SELECTIE ──────────────────────────────

def analyze_selection(selected_logs, feature_columns, n_features):
    counter = Counter()
    for selected in selected_logs:
        counter.update(selected)

    total_folds = len(selected_logs)

    print(f"\n  Numar fix de parametri selectati per fold: {n_features} "
          f"din {len(feature_columns)}")

    top_n = min(15, len(feature_columns))
    print(f"\n  Top {top_n} parametri cei mai frecvent selectati "
          f"(din {total_folds} fold-uri):")
    print(f"  {'Parametru':<35} {'Frecventa':>12}")
    print(f"  {'-'*48}")
    for feat, count in counter.most_common(top_n):
        pct = count / total_folds * 100
        print(f"  {feat:<35} {count:>4}/{total_folds} ({pct:>5.1f}%)")

    return counter


# ──────────────────────────── VIZUALIZARI ───────────────────────────────────

def plot_comparison(results_full, results_rfe, csv_name, n_features, output_dir, n_folds, n_repeats):
    output_dir.mkdir(parents=True, exist_ok=True)

    means  = [np.mean(results_full) * 100, np.mean(results_rfe) * 100]
    stds   = [np.std(results_full)  * 100, np.std(results_rfe)  * 100]
    labels = [f"SVM rbf\n(toti parametrii)", f"SVM rbf\n(RFE → {n_features} parametri)"]
    colors = ["steelblue", "seagreen"]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(labels, means, color=colors, width=0.5, yerr=stds, capsize=6)

    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                 f"{val:.1f}%", ha="center", va="bottom",
                 fontsize=13, fontweight="bold")

    ax.axhline(50, color="red", linestyle="--", linewidth=1.2, label="Random (50%)")
    ax.set_ylabel("Accuracy medie (%)", fontsize=12)
    ax.set_title(
        f"SVM rbf cu vs fara selectie RFE — {csv_name}\n"
        f"{n_repeats}x{n_folds}-Fold CV",
        fontsize=12
    )
    ax.set_ylim(0, 100)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out_path = output_dir / f"rfe_comparison_{csv_name}.png"
    plt.savefig(out_path, dpi=150)
    print(f"\n  Grafic comparatie salvat in {out_path}")


def plot_selection_frequency(counter, total_folds, csv_name, n_features, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    top_n = min(20, len(counter))
    items = counter.most_common(top_n)
    names = [x[0] for x in items]
    freqs = [x[1] / total_folds * 100 for x in items]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(len(names)), freqs, color="seagreen")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Frecventa selectie (%)")
    ax.set_title(
        f"Cei mai frecvent selectati parametri de RFE — {csv_name} "
        f"(Top {top_n}, tinta={n_features})",
        fontsize=12
    )
    ax.axhline(50, color="red", linestyle="--", linewidth=0.8,
               label="Selectat in >50% din fold-uri")
    ax.legend()
    plt.tight_layout()
    out_path = output_dir / f"rfe_frequency_{csv_name}.png"
    plt.savefig(out_path, dpi=150)
    print(f"  Grafic frecventa selectie salvat in {out_path}")


# ──────────────────────────── ENTRY POINT ───────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Selectie generica de parametri prin RFE, aplicabila pe orice CSV al proiectului SHG"
    )
    parser.add_argument("--csv", type=Path, required=True,
                         help="Calea catre CSV-ul de analizat")
    parser.add_argument("--n-features", type=int, required=True,
                         help="Numarul fix de parametri de pastrat prin RFE")
    parser.add_argument("--rfe-step", type=int, default=3,
                         help="Cati parametri elimina RFE la fiecare pas intern (implicit 3)")
    parser.add_argument("--n-repeats", type=int, default=25,
                         help="Numarul de repetari CV (implicit 25)")
    parser.add_argument("--n-folds", type=int, default=3,
                         help="Numarul de fold-uri CV (implicit 3)")
    parser.add_argument("--output-dir", type=Path, default=None,
                         help="Folder pentru grafice (implicit langa CSV-ul de input)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    csv_name   = args.csv.stem
    output_dir = args.output_dir or (args.csv.parent / f"plots_rfe_{csv_name}")

    print(f"\n{'='*60}")
    print(f"  Selectie generica de parametri — RFE")
    print(f"  CSV: {args.csv}")
    print(f"  Tinta: {args.n_features} parametri | Step: {args.rfe_step} | "
          f"{args.n_repeats}x{args.n_folds}-Fold CV")
    print(f"{'='*60}")

    df, feature_columns = load_data(args.csv)

    if len(feature_columns) == 0:
        print(f"\n[EROARE] Niciun parametru numeric detectat in CSV.")
        sys.exit(1)

    if args.n_features >= len(feature_columns):
        print(f"\n[ATENTIE] --n-features ({args.n_features}) >= numarul de "
              f"parametri detectati ({len(feature_columns)}). "
              f"RFE nu va elimina nimic — echivalent cu 'toti parametrii'.")

    # ── A) Referinta: SVM pe toti parametrii detectati ──────────────────────
    print(f"\n{'='*60}")
    print(f"  A) SVM rbf — toti cei {len(feature_columns)} parametri detectati")
    print(f"{'='*60}")
    results_full, yt_full, yp_full, _ = cross_validate(
        df, feature_columns, use_rfe=False,
        n_features=args.n_features, rfe_step=args.rfe_step,
        n_folds=args.n_folds, n_repeats=args.n_repeats,
        label="Toti parametrii"
    )

    # ── B) SVM pe parametrii selectati de RFE ───────────────────────────────
    print(f"\n{'='*60}")
    print(f"  B) SVM rbf — parametri selectati de RFE (tinta={args.n_features})")
    print(f"{'='*60}")
    results_rfe, yt_rfe, yp_rfe, selected_logs = cross_validate(
        df, feature_columns, use_rfe=True,
        n_features=args.n_features, rfe_step=args.rfe_step,
        n_folds=args.n_folds, n_repeats=args.n_repeats,
        label="RFE"
    )

    # ── Analiza selectiei ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Analiza parametrilor selectati de RFE")
    print(f"{'='*60}")
    counter = analyze_selection(selected_logs, feature_columns, args.n_features)

    # ── Sumar final ──────────────────────────────────────────────────────────
    mean_full = np.mean(results_full) * 100
    mean_rfe  = np.mean(results_rfe)  * 100

    print(f"\n{'='*60}")
    print(f"  REZULTATE FINALE — {csv_name}")
    print(f"{'='*60}")
    print(f"  {'Varianta':<32} {'Acc medie':>10} {'Std':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'-'*66}")
    print(f"  {'Toti parametrii ('+str(len(feature_columns))+')':<32} {mean_full:>9.1f}% "
          f"{np.std(results_full)*100:>7.1f}% "
          f"{min(results_full)*100:>7.1f}% {max(results_full)*100:>7.1f}%")
    print(f"  {'RFE ('+str(args.n_features)+')':<32} {mean_rfe:>9.1f}% "
          f"{np.std(results_rfe)*100:>7.1f}% "
          f"{min(results_rfe)*100:>7.1f}% {max(results_rfe)*100:>7.1f}%")
    print(f"\n  Diferenta (Toti - RFE)     : {mean_full - mean_rfe:+.1f} puncte procentuale")
    print(f"  Reducere dimensionalitate  : {len(feature_columns)} -> "
          f"{args.n_features} parametri "
          f"({(1 - args.n_features/len(feature_columns))*100:.0f}% reducere)")

    print(f"\n  Raport clasificare — toti parametrii:")
    print(classification_report(yt_full, yp_full, labels=["normal", "malign"], zero_division=0))

    print(f"\n  Raport clasificare — RFE:")
    print(classification_report(yt_rfe, yp_rfe, labels=["normal", "malign"], zero_division=0))

    print(f"\n  Generare grafice...")
    plot_comparison(results_full, results_rfe, csv_name, args.n_features,
                    output_dir, args.n_folds, args.n_repeats)
    plot_selection_frequency(counter, len(selected_logs), csv_name,
                             args.n_features, output_dir)

    plt.close("all")

    print(f"\n{'='*60}")
    print(f"  Gata! Graficele sunt in: {output_dir}")
    print(f"{'='*60}\n")