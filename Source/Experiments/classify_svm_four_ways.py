"""
classify_svm_four_ways.py
Comparatie SVM intre cele 4 metode propuse de profesor pentru
clasificarea normal/malign, folosind informatia din canalele SHG
forward (ch02) si backward (ch00):

    1. FORWARD singur   -- 64 parametri, doar imaginile ch02
    2. BACKWARD singur  -- 64 parametri, doar imaginile ch00
    3. F+B COMBINAT      -- 128 parametri (64 backward + 64 forward),
                            potrivite pe aceeasi achizitie fizica
    4. RAPORT R          -- 64 parametri, extrasi din imaginea noua
                            R = round(((F-B)/(F+B) + 1) * 128), clamp 255
                            (vezi features_R_ratio.py)

METODOLOGIE: toate cele 4 metode folosesc, la fiecare repetare, ACELASI
StratifiedKFold aplicat pe cele 18 grupuri de tesut normale/maligne --
un singur split reutilizat pentru toate cele 4 metode, identic cu
disciplina aplicata deja la comparatia canalelor
(classify_svm_channel_comparison.py). Filtrarea specifica fiecarei
metode (pe canal, pe perechi combinate, sau pe setul R) se aplica DOAR
dupa ce grupurile de train/test sunt deja fixate, garantand ca cele 4
metode sunt evaluate pe exact aceleasi impartiri de grupuri la fiecare
repetare -- comparatie corecta, fara nicio sursa suplimentara de
variatie intre metode.

Aceasta comparatie e primul pas al planului mai larg discutat cu
profesorul: fiecare din cele 4 metode va fi ulterior reevaluata si in
cadrul comparatiei HE vs non-HE (Sectiunea 10.1), ca experiment final
pentru toti algoritmii de clasificare.

Utilizare:
    python classify_svm_four_ways.py --features-csv "...\\features.csv" --r-csv "...\\features_R_ratio.csv"
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
from sklearn.metrics import accuracy_score, classification_report

sys.path.insert(0, str(Path(__file__).parent))


def _find_and_add_core(start: Path) -> bool:
    current = start
    for _ in range(5):
        candidate = current / "core"
        if (candidate / "config.py").exists():
            sys.path.insert(0, str(candidate))
            return True
        current = current.parent
    return False

_find_and_add_core(Path(__file__).resolve().parent)

DEFAULT_FEATURES_CSV = None
DEFAULT_R_CSV         = None
DEFAULT_PLOTS_DIR     = None
try:
    from config import CSV_FEATURES, CSV_DIR, PLOTS_DIR as CONFIG_PLOTS_DIR, DIAGNOSIS
    DEFAULT_FEATURES_CSV = CSV_FEATURES
    DEFAULT_R_CSV         = CSV_DIR / "features_R_ratio.csv"
    DEFAULT_PLOTS_DIR     = CONFIG_PLOTS_DIR / "svm_four_ways"
except ImportError:
    DIAGNOSIS = {
        "72066_1" : "normal", "72066_2" : "malign", "72066_3" : "normal",
        "83064_1" : "normal", "83064_2" : "malign", "83064_3" : "malign",
        "89376_1" : "malign", "89376_2" : "malign", "89376_3" : "displazic",
        "102312_1": "normal", "102312_2": "malign", "102312_3": "malign",
        "149113_1": "displazic", "149113_2": "displazic", "149113_3": "displazic",
        "155820_1": "malign", "155820_2": "malign", "155820_3": "displazic",
        "156222_1": "normal", "156222_2": "normal", "156222_3": "normal",
        "182467_1": "normal", "182467_3": "normal",
    }

# ──────────────────────────── CONFIGURARE ───────────────────────────────────

DIRECTION_PAIRS = ["0_180", "90_270", "45_225", "135_315"]
GLCM_PARAMS     = ["ASM", "Contrast", "Correlation", "IDM", "Entropy"]
DISTANCES       = [1, 10, 30]

N_FOLDS      = 3
N_REPEATS    = 25
KERNEL       = "rbf"
RANDOM_STATE = 42

CHANNEL_BACK    = "ch00"
CHANNEL_FORWARD = "ch02"

METHODS = ["forward", "backward", "combined", "R_ratio"]


def build_feature_columns(distances):
    return [
        "Mean", "StDev", "Skew", "Kurt",
    ] + [
        f"{param}_{pair}_d{dist}"
        for pair in DIRECTION_PAIRS
        for dist in distances
        for param in GLCM_PARAMS
    ]


BASE_FEATURE_COLUMNS = build_feature_columns(DISTANCES)


# ──────────────────────────── INCARCARE + PREGATIRE DATE ────────────────────

def add_diagnosis(df):
    df = df.copy()
    df["section_norm"] = df["section"].astype(str).str[0]
    df["probe_base"]   = df["probe"].astype(str).str.replace("HE", "", regex=False)
    df["key"]          = df["probe_base"] + "_" + df["section_norm"]
    df["diagnosis"]    = df["key"].map(DIAGNOSIS)
    df = df.dropna(subset=["diagnosis"])
    df = df[df["diagnosis"] != "displazic"]
    return df


def load_all_datasets(features_csv: Path, r_csv: Path):
    if not features_csv.exists():
        print(f"[EROARE] Nu gasesc {features_csv}")
        sys.exit(1)
    if not r_csv.exists():
        print(f"[EROARE] Nu gasesc {r_csv}. Ruleaza intai features_R_ratio.py")
        sys.exit(1)

    raw_df = pd.read_csv(features_csv)
    raw_df = add_diagnosis(raw_df)

    forward_df  = raw_df[raw_df["channel"] == CHANNEL_FORWARD].copy()
    backward_df = raw_df[raw_df["channel"] == CHANNEL_BACK].copy()

    # Combinat: potrivire pe aceeasi achizitie fizica (probe+section+polar)
    back_renamed    = backward_df.rename(columns={c: f"{c}_back" for c in BASE_FEATURE_COLUMNS})
    forward_renamed = forward_df.rename(columns={c: f"{c}_forward" for c in BASE_FEATURE_COLUMNS})
    merge_keys = ["probe", "section", "polar", "key", "diagnosis"]
    combined_df = pd.merge(
        back_renamed[merge_keys + [f"{c}_back" for c in BASE_FEATURE_COLUMNS]],
        forward_renamed[merge_keys + [f"{c}_forward" for c in BASE_FEATURE_COLUMNS]],
        on=merge_keys, how="inner"
    )
    combined_feature_columns = (
        [f"{c}_back" for c in BASE_FEATURE_COLUMNS] +
        [f"{c}_forward" for c in BASE_FEATURE_COLUMNS]
    )

    # R_ratio: CSV separat, adaugam diagnosis
    r_df = pd.read_csv(r_csv)
    r_df = add_diagnosis(r_df)

    datasets = {
        "forward" : (forward_df, BASE_FEATURE_COLUMNS),
        "backward": (backward_df, BASE_FEATURE_COLUMNS),
        "combined": (combined_df, combined_feature_columns),
        "R_ratio" : (r_df, BASE_FEATURE_COLUMNS),
    }

    print(f"\n  Perechi/imagini disponibile per metoda:")
    for name, (df, cols) in datasets.items():
        print(f"    {name:<10}: {len(df):>5} randuri | {df['key'].nunique()} grupuri | {len(cols)} parametri")

    return datasets


# ──────────────────────────── CROSS-VALIDATION UNIFICATA ────────────────────

def cross_validate_all_methods(datasets):
    """
    Grupurile de referinta (18 normale/maligne) sunt determinate din
    DIAGNOSIS direct -- ACELASI split e aplicat la toate cele 4 metode,
    la fiecare repetare, indiferent de cate randuri are fiecare dataset
    per grup.
    """
    all_keys    = sorted(DIAGNOSIS.keys())
    all_classes = np.array([DIAGNOSIS[k] for k in all_keys])
    mask        = all_classes != "displazic"
    keys        = np.array(all_keys)[mask]
    classes     = all_classes[mask]

    print(f"\n  Grupuri de referinta (normale+maligne): {len(keys)}")
    print(f"  Repetari: {N_REPEATS} | Fold-uri: {N_FOLDS}\n")

    results = {m: [] for m in METHODS}
    all_y_true = {m: [] for m in METHODS}
    all_y_pred = {m: [] for m in METHODS}

    for repeat in range(N_REPEATS):
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE + repeat)

        for train_idx, test_idx in skf.split(keys, classes):
            train_groups = keys[train_idx]
            test_groups  = keys[test_idx]

            for method in METHODS:
                df, feature_columns = datasets[method]

                train_df = df[df["key"].isin(train_groups)]
                test_df  = df[df["key"].isin(test_groups)]

                if train_df["diagnosis"].nunique() < 2 or len(test_df) == 0:
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

                results[method].append(accuracy_score(y_test, y_pred))
                all_y_true[method].extend(y_test.tolist())
                all_y_pred[method].extend(y_pred.tolist())

        if (repeat + 1) % 5 == 0:
            summary = " | ".join(f"{m}: {np.mean(results[m])*100:.1f}%" for m in METHODS)
            print(f"  Repetarea {repeat+1:2d}/{N_REPEATS} — {summary}")

    return results, all_y_true, all_y_pred


# ──────────────────────────── VIZUALIZARE ───────────────────────────────────

def plot_comparison(results, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = ["Forward\n(64 param.)", "Backward\n(64 param.)",
              "F+B combinat\n(128 param.)", "Raport R\n(64 param.)"]
    means  = [np.mean(results[m]) * 100 for m in METHODS]
    stds   = [np.std(results[m]) * 100 for m in METHODS]
    colors = ["darkorange", "steelblue", "seagreen", "mediumpurple"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, means, yerr=stds, capsize=6, color=colors)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.axhline(50, color="red", linestyle="--", linewidth=1, label="Random (50%)")
    ax.set_ylabel("Accuracy medie (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"SVM rbf — Comparatie 4 metode (Forward/Backward/Combinat/Raport R)\n"
                 f"{N_REPEATS}x{N_FOLDS}-Fold CV, acelasi split de grupuri pentru toate metodele")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "four_ways_comparison.png", dpi=150)
    print(f"\n  Grafic salvat in {output_dir / 'four_ways_comparison.png'}")


# ──────────────────────────── ENTRY POINT ───────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Comparatie SVM: Forward vs Backward vs F+B combinat vs Raport R"
    )
    parser.add_argument("--features-csv", type=Path, default=DEFAULT_FEATURES_CSV,
                         required=DEFAULT_FEATURES_CSV is None)
    parser.add_argument("--r-csv", type=Path, default=DEFAULT_R_CSV,
                         required=DEFAULT_R_CSV is None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PLOTS_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_dir = args.output_dir or (args.features_csv.parent.parent / "Plots" / "svm_four_ways")

    print(f"\n{'='*60}")
    print(f"  SVM {KERNEL} — Comparatie 4 metode (Forward/Backward/Combinat/Raport R)")
    print(f"{'='*60}")

    datasets = load_all_datasets(args.features_csv, args.r_csv)

    results, all_y_true, all_y_pred = cross_validate_all_methods(datasets)

    print(f"\n{'='*60}")
    print(f"  REZULTATE FINALE")
    print(f"{'='*60}")
    print(f"  {'Metoda':<15} {'Nr. param.':>12} {'Accuracy medie':>16} {'Std':>8}")
    print(f"  {'-'*54}")
    n_params = {"forward": 64, "backward": 64, "combined": 128, "R_ratio": 64}
    for m in METHODS:
        print(f"  {m:<15} {n_params[m]:>12} {np.mean(results[m])*100:>15.1f}% {np.std(results[m])*100:>7.1f}%")

    for m in METHODS:
        print(f"\n  Raport clasificare — {m}:")
        print(classification_report(all_y_true[m], all_y_pred[m],
                                     labels=["normal", "malign"], zero_division=0))

    plot_comparison(results, output_dir)
    plt.close("all")

    print(f"\n{'='*60}")
    print(f"  Gata! Grafic salvat in: {output_dir}")
    print(f"{'='*60}\n")