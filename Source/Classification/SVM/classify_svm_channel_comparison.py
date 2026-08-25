"""
classify_svm_channel_comparison.py
Comparatie SVM intre canalele SHG BACKWARD (ch00) si FORWARD (ch02),
la solicitarea profesorului: cat de bine se clasifica normal/malign
folosind fiecare canal separat, si cat de bine generalizeaza un model
antrenat pe un canal atunci cand e testat pe celalalt.

Patru scenarii evaluate:
    back  -> back      (antrenat pe ch00, testat pe ch00)
    forward -> forward (antrenat pe ch02, testat pe ch02)
    back  -> forward   (antrenat pe ch00, testat pe ch02)
    forward -> back    (antrenat pe ch02, testat pe ch00)

METODOLOGIE CORECTATA (identica cu fix-ul din Sectiunea 10.1, aplicat
initial pentru compararea HE vs non-HE): se foloseste UN SINGUR
StratifiedKFold per repetare, aplicat pe cele 18 grupuri de tesut
unice. Toate cele 4 scenarii ruleaza in interiorul aceluiasi loop de
fold, refolosind literal aceiasi train_groups/test_groups -- filtrarea
pe canal (ch00/ch02) se aplica DUPA stabilirea grupurilor de train/test,
nu inainte. Aceasta garanteaza 0% leakage prin constructie: nu pot
exista doua obiecte StratifiedKFold independente actionand asupra
acelorasi 18 grupuri, care ar putea plasa acelasi grup de tesut
simultan in train (pt un canal) si in test (pt celalalt canal).

Utilizare:
    python classify_svm_channel_comparison.py
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

# DIAGNOSIS si un CSV/PLOTS_DIR implicit (folosite doar daca --csv nu e dat)
DEFAULT_CSV_PATH  = None
DEFAULT_PLOTS_DIR = None
try:
    from config import CSV_FEATURES, PLOTS_DIR as CONFIG_PLOTS_DIR, DIAGNOSIS
    DEFAULT_CSV_PATH  = CSV_FEATURES
    DEFAULT_PLOTS_DIR = CONFIG_PLOTS_DIR / "svm_channel_comparison"
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

# Conventie canale (utils.py): ch00 = SHG backward, ch02 = SHG forward
CHANNEL_BACK    = "ch00"
CHANNEL_FORWARD = "ch02"

SCENARIOS = [
    ("back",    "back"),
    ("forward", "forward"),
    ("back",    "forward"),
    ("forward", "back"),
]
CHANNEL_MAP = {"back": CHANNEL_BACK, "forward": CHANNEL_FORWARD}


def build_feature_columns(distances):
    return [
        "Mean", "StDev", "Skew", "Kurt",
    ] + [
        f"{param}_{pair}_d{dist}"
        for pair in DIRECTION_PAIRS
        for dist in distances
        for param in GLCM_PARAMS
    ]


FEATURE_COLUMNS = build_feature_columns(DISTANCES)


# ──────────────────────────── INCARCARE DATE ────────────────────────────────

def load_data(csv_path: Path):
    print(f"\n{'='*60}")
    print(f"  SVM {KERNEL} — Comparatie canal SHG (backward ch00 vs forward ch02)")
    print(f"  {N_REPEATS}x{N_FOLDS}-Fold CV | Distante GLCM: {DISTANCES}")
    print(f"{'='*60}\n")

    if not csv_path.exists():
        print(f"[EROARE] Nu gasesc CSV-ul la: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"  Total imagini incarcate : {len(df)}")

    df["section_norm"] = df["section"].astype(str).str[0]
    df["probe_base"]   = df["probe"].str.replace("HE", "", regex=False)
    df["key"]          = df["probe_base"] + "_" + df["section_norm"]
    df["diagnosis"]    = df["key"].map(DIAGNOSIS)

    df = df.dropna(subset=["diagnosis"])
    df = df[df["diagnosis"] != "displazic"]

    print(f"  Normal    : {(df['diagnosis']=='normal').sum()} imagini")
    print(f"  Malign    : {(df['diagnosis']=='malign').sum()} imagini")
    print(f"  ch00 (back)    : {(df['channel']==CHANNEL_BACK).sum()} imagini")
    print(f"  ch02 (forward) : {(df['channel']==CHANNEL_FORWARD).sum()} imagini")

    return df


# ──────────────────────────── CROSS-VALIDATION CORECTATA ────────────────────

def cross_validate(df):
    """
    UN SINGUR StratifiedKFold per repetare, pe cele 18 grupuri unice.
    Toate cele 4 scenarii (back->back, forward->forward, back->forward,
    forward->back) refolosesc EXACT aceiasi train_keys/test_keys la
    fiecare fold -- filtrarea pe canal se aplica DOAR dupa alegerea
    grupurilor, niciodata inainte.
    """
    probe_sections = (
        df.groupby("key")["diagnosis"]
        .agg(lambda x: x.value_counts().index[0])
        .reset_index()
        .rename(columns={"diagnosis": "dominant_class"})
    )
    keys    = probe_sections["key"].values
    classes = probe_sections["dominant_class"].values

    print(f"\n  Sectiuni unice : {len(keys)}")
    print(f"  Normal         : {(classes=='normal').sum()}")
    print(f"  Malign         : {(classes=='malign').sum()}")
    print(f"  Repetari       : {N_REPEATS}")
    print(f"  Fold-uri       : {N_FOLDS}\n")

    # rezultate[scenariu] = lista de accuracy-uri
    results = {f"{train_ch}->{test_ch}": [] for train_ch, test_ch in SCENARIOS}
    all_y_true = {k: [] for k in results}
    all_y_pred = {k: [] for k in results}

    for repeat in range(N_REPEATS):
        # UN SINGUR split, folosit pentru toate cele 4 scenarii ale acestei repetari
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                               random_state=RANDOM_STATE + repeat)

        for train_idx, test_idx in skf.split(keys, classes):
            train_groups = keys[train_idx]
            test_groups  = keys[test_idx]

            for train_ch_label, test_ch_label in SCENARIOS:
                train_channel = CHANNEL_MAP[train_ch_label]
                test_channel  = CHANNEL_MAP[test_ch_label]

                train_df = df[
                    df["key"].isin(train_groups) & (df["channel"] == train_channel)
                ]
                test_df = df[
                    df["key"].isin(test_groups) & (df["channel"] == test_channel)
                ]

                if train_df["diagnosis"].nunique() < 2 or len(test_df) == 0:
                    continue

                X_train = train_df[FEATURE_COLUMNS].astype(float)
                y_train = train_df["diagnosis"]
                X_test  = test_df[FEATURE_COLUMNS].astype(float)
                y_test  = test_df["diagnosis"]

                scaler  = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test  = scaler.transform(X_test)

                model = SVC(kernel=KERNEL, class_weight="balanced", random_state=RANDOM_STATE)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                key = f"{train_ch_label}->{test_ch_label}"
                results[key].append(accuracy_score(y_test, y_pred))
                all_y_true[key].extend(y_test.tolist())
                all_y_pred[key].extend(y_pred.tolist())

        if (repeat + 1) % 5 == 0:
            summary = " | ".join(f"{k}: {np.mean(v)*100:.1f}%" for k, v in results.items())
            print(f"  Repetarea {repeat+1:2d}/{N_REPEATS} — {summary}")

    return results, all_y_true, all_y_pred


# ──────────────────────────── VIZUALIZARI ───────────────────────────────────

def plot_comparison(results, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = list(results.keys())
    means  = [np.mean(results[k]) * 100 for k in labels]
    stds   = [np.std(results[k]) * 100 for k in labels]
    colors = ["steelblue", "steelblue", "darkorange", "darkorange"]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(labels, means, yerr=stds, capsize=6, color=colors)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.axhline(50, color="red", linestyle="--", linewidth=1, label="Random (50%)")
    ax.set_ylabel("Accuracy medie (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"SVM {KERNEL} — Backward (ch00) vs Forward (ch02)\n"
                 f"{N_REPEATS}x{N_FOLDS}-Fold CV, split unic per repetare (fara leakage)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "channel_comparison.png", dpi=150)
    print(f"\n  Grafic salvat in {output_dir / 'channel_comparison.png'}")


# ──────────────────────────── ENTRY POINT ───────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Comparatie SVM intre canalele SHG backward (ch00) si forward (ch02)"
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH,
                         required=DEFAULT_CSV_PATH is None,
                         help="Calea catre CSV-ul de analizat (ex. features.csv)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PLOTS_DIR,
                         help="Folder pentru grafic (implicit: Results/Plots/svm_channel_comparison)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    output_dir = args.output_dir or (args.csv.parent.parent / "Plots" / "svm_channel_comparison")

    df = load_data(args.csv)

    results, all_y_true, all_y_pred = cross_validate(df)

    print(f"\n{'='*60}")
    print(f"  REZULTATE FINALE — Backward (ch00) vs Forward (ch02)")
    print(f"{'='*60}")
    print(f"  {'Scenariu':<20} {'Accuracy medie':>15} {'Std':>8}")
    print(f"  {'-'*45}")
    for key, vals in results.items():
        print(f"  {key:<20} {np.mean(vals)*100:>14.1f}% {np.std(vals)*100:>7.1f}%")

    print(f"\n  Interpretare:")
    print(f"  back->back si forward->forward   = performanta 'in-domeniu' pe fiecare canal")
    print(f"  back->forward si forward->back   = capacitatea de generalizare intre canale")

    for key in results:
        print(f"\n  Raport clasificare — {key}:")
        print(classification_report(all_y_true[key], all_y_pred[key],
                                     labels=["normal", "malign"], zero_division=0))

    plot_comparison(results, output_dir)
    plt.close("all")

    print(f"\n{'='*60}")
    print(f"  Gata! Grafic salvat in: {output_dir}")
    print(f"{'='*60}\n")