"""
test_distance_combinations.py
Testeaza TOATE combinatiile posibile de 3 distante GLCM alese din
{1, 5, 10, 15, 20, 25, 30} -- C(7,3) = 35 combinatii -- pentru a
verifica riguros daca d1+d10+d30 este intr-adevar configuratia optima
identificata initial, sau daca exista o combinatie mai buna.

Foloseste Results/CSV/features_multi_distance.csv (generat de
main_multi_distance.py), care contine GLCM calculat separat la
fiecare din cele 7 distante. Pentru fiecare combinatie, selecteaza
doar coloanele corespunzatoare celor 3 distante alese si ruleaza
EXACT aceeasi metodologie SVM rbf 25x3-Fold CV (Sectiunea 5.2) ca
in classify_svm_kfold.py, la nivel de grup de tesut (probe+sectiune).

Output:
    - Tabel complet cu toate cele 35 combinatii, sortat descrescator
      dupa accuracy medie
    - Grafic bar chart cu top 10 combinatii
    - CSV cu rezultatele complete (distance_combinations_results.csv)

Utilizare:
    python test_distance_combinations.py
    python test_distance_combinations.py --n-repeats 10   (rulare rapida de test)
"""

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

def _find_and_add_core(start: Path) -> bool:
    """Urca din folderul scriptului pana gaseste un folder 'core' cu config.py."""
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
    from config import CSV_DIR, PLOTS_DIR, DIAGNOSIS
    INPUT_CSV  = CSV_DIR / "features_multi_distance.csv"
    OUTPUT_DIR = PLOTS_DIR / "distance_combinations"
except ImportError:
    INPUT_CSV  = Path(r"C:\Users\dimas\Downloads\projects\venv\Results\CSV\features_multi_distance.csv")
    OUTPUT_DIR = Path(r"C:\Users\dimas\Downloads\projects\venv\Results\Plots\distance_combinations")
    DIAGNOSIS = {
        "72066_1": "normal", "72066_2": "malign", "72066_3": "normal",
        "83064_1": "normal", "83064_2": "malign", "83064_3": "malign",
        "89376_1": "malign", "89376_2": "malign", "89376_3": "displazic",
        "102312_1": "normal", "102312_2": "malign", "102312_3": "malign",
        "149113_1": "displazic", "149113_2": "displazic", "149113_3": "displazic",
        "155820_1": "malign", "155820_2": "malign", "155820_3": "displazic",
        "156222_1": "normal", "156222_2": "normal", "156222_3": "normal",
        "182467_1": "normal", "182467_3": "normal",
    }

# ──────────────────────────── CONFIGURARE ───────────────────────────────────

ALL_DISTANCES   = [1, 5, 10, 15, 20, 25, 30]
DIRECTION_PAIRS = ["0_180", "90_270", "45_225", "135_315"]
GLCM_PARAMS     = ["ASM", "Contrast", "Correlation", "IDM", "Entropy"]

KERNEL       = "rbf"
RANDOM_STATE = 42
N_FOLDS      = 3


# ──────────────────────────── COLOANE PENTRU O COMBINATIE ───────────────────

def build_feature_columns(distances):
    """Identic cu build_feature_columns din classify_svm_kfold.py."""
    return [
        "Mean", "StDev", "Skew", "Kurt",
    ] + [
        f"{param}_{pair}_d{dist}"
        for pair in DIRECTION_PAIRS
        for dist in distances
        for param in GLCM_PARAMS
    ]


# ──────────────────────────── INCARCARE DATE ────────────────────────────────

def load_data(csv_path: Path):
    if not csv_path.exists():
        print(f"[EROARE] Nu gasesc CSV-ul la: {csv_path}")
        print(f"  Genereaza-l intai cu: python main_multi_distance.py")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    df["section_norm"] = df["section"].astype(str).str[0]
    df["probe_base"]   = df["probe"].astype(str).str.replace("HE", "", regex=False)
    df["key"]          = df["probe_base"] + "_" + df["section_norm"]
    df["diagnosis"]    = df["key"].map(DIAGNOSIS)

    df = df.dropna(subset=["diagnosis"])
    df = df[df["diagnosis"] != "displazic"]

    print(f"  CSV incarcat: {csv_path.name}")
    print(f"  Total imagini: {len(df)} | "
          f"normal: {(df['diagnosis']=='normal').sum()} | "
          f"malign: {(df['diagnosis']=='malign').sum()}")

    return df


# ──────────────────────────── CROSS-VALIDATION ──────────────────────────────

def cross_validate(df, feature_columns, n_repeats):
    """
    Stratified 3-Fold CV repetat n_repeats ori, la nivel de grup de
    tesut (probe+sectiune) -- identic metodologic cu classify_svm_kfold.py.
    """
    probe_sections = (
        df.groupby("key")["diagnosis"]
        .agg(lambda x: x.value_counts().index[0])
        .reset_index()
        .rename(columns={"diagnosis": "dominant_class"})
    )
    keys    = probe_sections["key"].values
    classes = probe_sections["dominant_class"].values

    results = []

    for repeat in range(n_repeats):
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                               random_state=RANDOM_STATE + repeat)

        for train_idx, test_idx in skf.split(keys, classes):
            train_keys = keys[train_idx]
            test_keys  = keys[test_idx]

            train_df = df[df["key"].isin(train_keys)]
            test_df  = df[df["key"].isin(test_keys)]

            if train_df["diagnosis"].nunique() < 2 or len(test_df) == 0:
                continue

            X_train = train_df[feature_columns].astype(float)
            y_train = train_df["diagnosis"]
            X_test  = test_df[feature_columns].astype(float)
            y_test  = test_df["diagnosis"]

            scaler  = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test  = scaler.transform(X_test)

            model = SVC(kernel=KERNEL, class_weight="balanced",
                        random_state=RANDOM_STATE)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            results.append(accuracy_score(y_test, y_pred))

    return results


# ──────────────────────────── TESTARE TOATE COMBINATIILE ────────────────────

def test_all_combinations(df, n_repeats):
    combos = list(combinations(ALL_DISTANCES, 3))
    print(f"\n  Testare {len(combos)} combinatii posibile de 3 distante "
          f"din {ALL_DISTANCES}")
    print(f"  {n_repeats}x{N_FOLDS}-Fold CV per combinatie "
          f"({n_repeats * N_FOLDS} evaluari x {len(combos)} combinatii = "
          f"{n_repeats * N_FOLDS * len(combos)} antrenari SVM totale)\n")

    rows = []
    for combo in tqdm(combos, desc="Combinatii distante", unit="combo"):
        feature_columns = build_feature_columns(list(combo))
        results = cross_validate(df, feature_columns, n_repeats)

        rows.append({
            "distante"      : "+".join(f"d{d}" for d in combo),
            "d1"            : combo[0],
            "d2"            : combo[1],
            "d3"            : combo[2],
            "accuracy_medie": np.mean(results) * 100,
            "accuracy_std"  : np.std(results) * 100,
            "accuracy_min"  : np.min(results) * 100,
            "accuracy_max"  : np.max(results) * 100,
        })

    results_df = pd.DataFrame(rows).sort_values(
        "accuracy_medie", ascending=False
    ).reset_index(drop=True)

    return results_df


# ──────────────────────────── VIZUALIZARE ───────────────────────────────────

def plot_top_combinations(results_df, output_dir, top_n=10):
    output_dir.mkdir(parents=True, exist_ok=True)
    top = results_df.head(top_n)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["darkorange" if row["distante"] == "d1+d10+d30" else "steelblue"
              for _, row in top.iterrows()]

    bars = ax.bar(top["distante"], top["accuracy_medie"],
                   yerr=top["accuracy_std"], color=colors, capsize=4)

    for bar, val in zip(bars, top["accuracy_medie"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=9,
                 fontweight="bold")

    ax.set_ylabel("Accuracy medie (%)")
    ax.set_title(f"Top {top_n} combinatii de distante GLCM\n"
                 f"(portocaliu = configuratia standard d1+d10+d30)")
    ax.set_xticklabels(top["distante"], rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out_path = output_dir / "top_distance_combinations.png"
    plt.savefig(out_path, dpi=150)
    print(f"\n  Grafic salvat in {out_path}")


# ──────────────────────────── ENTRY POINT ───────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Testeaza toate combinatiile de 3 distante GLCM din {1,5,10,15,20,25,30}"
    )
    parser.add_argument("--csv", type=Path, default=INPUT_CSV,
                         help="CSV cu GLCM la toate distantele individuale")
    parser.add_argument("--n-repeats", type=int, default=25,
                         help="Repetari CV per combinatie (implicit 25; "
                              "foloseste o valoare mica ex. 5-10 pentru test rapid)")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"  Testare exhaustiva combinatii distante GLCM")
    print(f"{'='*60}")

    df = load_data(args.csv)
    results_df = test_all_combinations(df, args.n_repeats)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_csv_path = args.output_dir / "distance_combinations_results.csv"
    results_df.to_csv(results_csv_path, index=False)

    print(f"\n{'='*60}")
    print(f"  REZULTATE — Top 10 combinatii")
    print(f"{'='*60}")
    print(results_df.head(10).to_string(index=False))

    standard_row = results_df[results_df["distante"] == "d1+d10+d30"]
    if not standard_row.empty:
        rank = results_df.index[results_df["distante"] == "d1+d10+d30"][0] + 1
        print(f"\n  Configuratia standard d1+d10+d30 se claseaza pe "
              f"locul {rank}/{len(results_df)}")
        print(f"  Accuracy: {standard_row['accuracy_medie'].values[0]:.1f}%")
        best_acc = results_df.iloc[0]["accuracy_medie"]
        diff = best_acc - standard_row["accuracy_medie"].values[0]
        print(f"  Diferenta fata de cea mai buna combinatie: {diff:+.1f} puncte procentuale")

    plot_top_combinations(results_df, args.output_dir)

    print(f"\n{'='*60}")
    print(f"  Rezultate complete salvate in: {results_csv_path}")
    print(f"{'='*60}\n")