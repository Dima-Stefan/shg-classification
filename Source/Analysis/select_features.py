"""
select_features.py
Script generic si reutilizabil de selectie de parametri prin Lasso
(LogisticRegression cu penalizare L1), aplicabil pe orice CSV generat
de pipeline-ul proiectului (features.csv, features_glrlm.csv,
features_corr_std.csv, features_patches_*.csv etc.) fara sa fie nevoie
de un script de clasificare nou pentru fiecare combinatie.

Coloanele de parametri sunt detectate AUTOMAT: toate coloanele
numerice din CSV, EXCEPTAND coloanele de metadate cunoscute.

Prin argumentul --group se poate restricționa analiza la un subset
de parametri extras direct din CSV, fara sa fie nevoie de un CSV nou:
    all    -> toti parametrii detectati (implicit)
    glcm   -> histograma (4) + GLCM (60)  = 64 parametri
    glrlm  -> histograma (4) + GLRLM (20) = 24 parametri

Metodologie identica cu classify_svm_kfold_lasso.py: selectia Lasso
se aplica STRICT in interiorul fiecarui fold de CV, folosind doar
datele de training ale acelui fold, pentru a evita orice leakage.

Utilizare:
    python select_features.py --csv "C:\\...\\Results\\features_glrlm.csv"
    python select_features.py --csv "C:\\...\\Results\\features_glrlm.csv" --group glrlm
    python select_features.py --csv "C:\\...\\Results\\features_glrlm.csv" --group glcm
    python select_features.py --csv "C:\\...\\Results\\features_glrlm.csv" --lasso-c 0.3

Argumente:
    --csv         Calea catre CSV-ul de analizat (OBLIGATORIU)
    --group       Subsetul de parametri: all | glcm | glrlm (implicit: all)
    --lasso-c     Forta de regularizare Lasso (implicit 0.5)
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


# ──────────────────────────── GRUPURI DE PARAMETRI ─────────────────────────

# Prefixele care identifica fiecare grup — folosite pentru filtrarea
# automata a coloanelor din CSV prin argumentul --group
GLCM_PARAM_NAMES  = ["ASM", "Contrast", "Correlation", "IDM", "Entropy"]
GLRLM_PARAM_NAMES = ["SRE", "LRE", "GLN", "RLN", "RP"]
HISTOGRAM_NAMES   = ["Mean", "StDev", "Skew", "Kurt"]

DIRECTION_PAIRS = ["0_180", "90_270", "45_225", "135_315"]
DISTANCES       = [1, 10, 30]


def filter_columns_by_group(all_columns: list, group: str) -> list:
    """
    Filtreaza lista de coloane detectate automat, pastrand doar
    parametrii din grupul specificat.

    group='all'   -> toti parametrii (nicio filtrare)
    group='glcm'  -> histograma + GLCM (parametrii cu prefix
                     ASM/Contrast/Correlation/IDM/Entropy)
    group='glrlm' -> histograma + GLRLM (parametrii cu prefix
                     SRE/LRE/GLN/RLN/RP)

    Histograma (Mean, StDev, Skew, Kurt) e inclusa in toate grupurile
    ca baza, pentru ca orice clasificare are nevoie de ea.
    """
    if group == "all":
        return all_columns

    # Parametrii histograma — intotdeauna inclusi
    hist_cols = [c for c in all_columns if c in HISTOGRAM_NAMES]

    if group == "glcm":
        # Coloane care incep cu un prefix GLCM cunoscut
        glcm_cols = [
            c for c in all_columns
            if any(c.startswith(p + "_") for p in GLCM_PARAM_NAMES)
        ]
        selected = hist_cols + glcm_cols
    elif group == "glrlm":
        # Coloane care incep cu un prefix GLRLM cunoscut
        glrlm_cols = [
            c for c in all_columns
            if any(c.startswith(p + "_") for p in GLRLM_PARAM_NAMES)
        ]
        selected = hist_cols + glrlm_cols
    else:
        print(f"[EROARE] Grup necunoscut: '{group}'. "
              f"Valori acceptate: all, glcm, glrlm.")
        sys.exit(1)

    if not selected:
        print(f"[EROARE] Niciun parametru gasit pentru grupul '{group}' "
              f"in CSV. Verifica daca CSV-ul contine parametrii corecti.")
        sys.exit(1)

    return selected


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


# ──────────────────────────── SELECTIE LASSO ────────────────────────────────

def select_features_lasso(X_train_scaled, y_train, feature_names, lasso_c):
    """
    Antreneaza LogisticRegression cu penalizare L1 pe datele de train
    (deja scalate) si returneaza lista parametrilor cu coeficient != 0.
    Identic metodologic cu classify_svm_kfold_lasso.py.
    """
    lasso = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        C=lasso_c,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        max_iter=5000,
    )
    lasso.fit(X_train_scaled, y_train)

    coefs = lasso.coef_[0]
    selected = [feature_names[i] for i in range(len(coefs)) if coefs[i] != 0]

    if len(selected) == 0:
        best_idx = np.argmax(np.abs(coefs))
        selected = [feature_names[best_idx]]

    return selected


# ──────────────────────────── CROSS-VALIDATION ──────────────────────────────

def cross_validate(df, feature_columns, use_lasso, lasso_c, n_folds, n_repeats, label):
    """
    Stratified n_folds-Fold CV repetat n_repeats ori la nivel de
    proba+sectiune. Identic metodologic cu classify_svm_kfold_lasso.py,
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

            if use_lasso:
                selected = select_features_lasso(
                    X_train_full_scaled, y_train, feature_columns, lasso_c
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

def analyze_selection(selected_logs, feature_columns):
    counter = Counter()
    sizes   = []

    for selected in selected_logs:
        sizes.append(len(selected))
        counter.update(selected)

    total_folds = len(selected_logs)

    print(f"\n  Numar mediu de parametri selectati per fold: "
          f"{np.mean(sizes):.1f} din {len(feature_columns)}")
    print(f"  Min: {min(sizes)} | Max: {max(sizes)}")

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

def plot_comparison(results_full, results_lasso, csv_name, output_dir, n_folds, n_repeats):
    output_dir.mkdir(parents=True, exist_ok=True)

    means  = [np.mean(results_full) * 100, np.mean(results_lasso) * 100]
    stds   = [np.std(results_full)  * 100, np.std(results_lasso)  * 100]
    n_full = len(results_full)
    labels = [f"SVM rbf\n(toti parametrii)", "SVM rbf\n(selectati Lasso)"]
    colors = ["steelblue", "darkorange"]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(labels, means, color=colors, width=0.5, yerr=stds, capsize=6)

    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                 f"{val:.1f}%", ha="center", va="bottom",
                 fontsize=13, fontweight="bold")

    ax.axhline(50, color="red", linestyle="--", linewidth=1.2, label="Random (50%)")
    ax.set_ylabel("Accuracy medie (%)", fontsize=12)
    ax.set_title(
        f"SVM rbf cu vs fara selectie Lasso — {csv_name}\n"
        f"{n_repeats}x{n_folds}-Fold CV",
        fontsize=12
    )
    ax.set_ylim(0, 100)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out_path = output_dir / f"lasso_comparison_{csv_name}.png"
    plt.savefig(out_path, dpi=150)
    print(f"\n  Grafic comparatie salvat in {out_path}")


def plot_selection_frequency(counter, total_folds, csv_name, output_dir):
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
    ax.set_title(f"Cei mai frecvent selectati parametri de Lasso — {csv_name} (Top {top_n})",
                 fontsize=12)
    ax.axhline(50, color="red", linestyle="--", linewidth=0.8,
               label="Selectat in >50% din fold-uri")
    ax.legend()
    plt.tight_layout()
    out_path = output_dir / f"lasso_frequency_{csv_name}.png"
    plt.savefig(out_path, dpi=150)
    print(f"  Grafic frecventa selectie salvat in {out_path}")


# ──────────────────────────── ENTRY POINT ───────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Selectie generica de parametri prin Lasso, aplicabila pe orice CSV al proiectului SHG"
    )
    parser.add_argument("--csv", type=Path, required=True,
                         help="Calea catre CSV-ul de analizat")
    parser.add_argument("--group", type=str, default="all",
                         choices=["all", "glcm", "glrlm"],
                         help="Subsetul de parametri: all | glcm | glrlm (implicit: all)")
    parser.add_argument("--lasso-c", type=float, default=0.5,
                         help="Forta de regularizare Lasso (implicit 0.5)")
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
    group_tag  = f"_{args.group}" if args.group != "all" else ""
    output_dir = args.output_dir or (args.csv.parent / f"plots_lasso_{csv_name}{group_tag}")

    print(f"\n{'='*60}")
    print(f"  Selectie generica de parametri — Lasso")
    print(f"  CSV: {args.csv}")
    print(f"  Grup: {args.group} | Lasso C = {args.lasso_c} | "
          f"{args.n_repeats}x{args.n_folds}-Fold CV")
    print(f"{'='*60}")

    df, all_feature_columns = load_data(args.csv)

    if len(all_feature_columns) == 0:
        print(f"\n[EROARE] Niciun parametru numeric detectat in CSV.")
        sys.exit(1)

    # Filtram coloanele dupa grupul cerut
    feature_columns = filter_columns_by_group(all_feature_columns, args.group)

    if args.group != "all":
        print(f"\n  Filtrare activa — grup '{args.group}':")
        print(f"  Parametri disponibili in CSV : {len(all_feature_columns)}")
        print(f"  Parametri folositi in analiza: {len(feature_columns)}")

    # ── A) Referinta: SVM pe parametrii grupului ─────────────────────────────
    print(f"\n{'='*60}")
    print(f"  A) SVM rbf — toti cei {len(feature_columns)} parametri "
          f"(grup: {args.group})")
    print(f"{'='*60}")
    results_full, yt_full, yp_full, _ = cross_validate(
        df, feature_columns, use_lasso=False, lasso_c=args.lasso_c,
        n_folds=args.n_folds, n_repeats=args.n_repeats, label="Toti parametrii"
    )

    # ── B) SVM pe parametrii selectati de Lasso ─────────────────────────────
    print(f"\n{'='*60}")
    print(f"  B) SVM rbf — parametri selectati de Lasso (C={args.lasso_c})")
    print(f"{'='*60}")
    results_lasso, yt_lasso, yp_lasso, selected_logs = cross_validate(
        df, feature_columns, use_lasso=True, lasso_c=args.lasso_c,
        n_folds=args.n_folds, n_repeats=args.n_repeats, label="Lasso"
    )

    # ── Analiza selectiei ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Analiza parametrilor selectati de Lasso")
    print(f"{'='*60}")
    counter, sizes = analyze_selection(selected_logs, feature_columns)

    # ── Sumar final ──────────────────────────────────────────────────────────
    mean_full  = np.mean(results_full)  * 100
    mean_lasso = np.mean(results_lasso) * 100

    print(f"\n{'='*60}")
    print(f"  REZULTATE FINALE — {csv_name} (grup: {args.group})")
    print(f"{'='*60}")
    print(f"  {'Varianta':<30} {'Acc medie':>10} {'Std':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'-'*66}")
    print(f"  {'Toti parametrii ('+str(len(feature_columns))+')':<30} {mean_full:>9.1f}% "
          f"{np.std(results_full)*100:>7.1f}% "
          f"{min(results_full)*100:>7.1f}% {max(results_full)*100:>7.1f}%")
    print(f"  {'Selectati Lasso (~'+str(round(np.mean(sizes)))+')':<30} {mean_lasso:>9.1f}% "
          f"{np.std(results_lasso)*100:>7.1f}% "
          f"{min(results_lasso)*100:>7.1f}% {max(results_lasso)*100:>7.1f}%")
    print(f"\n  Diferenta (Toti - Lasso): {mean_full - mean_lasso:+.1f} puncte procentuale")
    print(f"  Reducere dimensionalitate: {len(feature_columns)} -> "
          f"~{round(np.mean(sizes))} parametri "
          f"({(1 - np.mean(sizes)/len(feature_columns))*100:.0f}% reducere)")

    print(f"\n  Raport clasificare — toti parametrii:")
    print(classification_report(yt_full, yp_full, labels=["normal", "malign"], zero_division=0))

    print(f"\n  Raport clasificare — Lasso:")
    print(classification_report(yt_lasso, yp_lasso, labels=["normal", "malign"], zero_division=0))

    print(f"\n  Generare grafice...")
    csv_name_group = f"{csv_name}{group_tag}"
    plot_comparison(results_full, results_lasso, csv_name_group, output_dir,
                    args.n_folds, args.n_repeats)
    plot_selection_frequency(counter, len(selected_logs), csv_name_group, output_dir)

    plt.close("all")

    print(f"\n{'='*60}")
    print(f"  Gata! Graficele sunt in: {output_dir}")
    print(f"{'='*60}\n")