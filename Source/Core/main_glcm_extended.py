"""
main_glcm_extended.py
Combina cei 85 de parametri existenti (histograma + GLCM + GLRLM + CAF,
din features_caf.csv) cu cei 108 parametri GLCM extinsi noi, rezultand
193 de parametri in total.

Salveaza in Results/features_glcm_extended.csv.

Parametrii noi (9 parametri x 4 directii x 3 distante = 108):
    Grup 1: Autocorrelation, Dissimilarity, Homogeneity,
            MaxProbability, SumOfSquares
    Grup 2: SumAverage, SumEntropy, DifferenceVariance, DifferenceEntropy

Utilizare:
    python main_glcm_extended.py
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from features_glcm_extended import extract_glcm_extended_features


# ──────────────────────────── CALE IMPLICITA ────────────────────────────────

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
from config import CSV_FEATURES_CAF as DEFAULT_INPUT_CSV, CSV_FEATURES_GLCM_EXT as DEFAULT_OUTPUT_CSV


# ──────────────────────────── PROCESARE ─────────────────────────────────────

def process_all(input_csv: Path, output_csv: Path) -> None:
    print(f"\n{'='*60}")
    print(f"  SHG Feature Extraction — Adaugare GLCM Extins (108 param)")
    print(f"{'='*60}")
    print(f"  Input  : {input_csv}")
    print(f"  Output : {output_csv}")
    print(f"{'='*60}\n")

    if not input_csv.exists():
        print(f"[EROARE] Nu gasesc CSV-ul la: {input_csv}")
        sys.exit(1)

    df = pd.read_csv(input_csv)
    print(f"  Randuri incarcate: {len(df)} | Parametri existenti: {len(df.columns) - 6}")

    if "filepath" not in df.columns:
        print(f"[EROARE] CSV-ul nu contine coloana 'filepath'.")
        sys.exit(1)

    # Calculam parametrii GLCM extinsi pentru fiecare imagine
    all_extended = []
    errors       = []

    for _, row in tqdm(df.iterrows(), desc="Calculare GLCM extins",
                       total=len(df), unit="img"):
        result = extract_glcm_extended_features(row["filepath"])
        if result is None:
            errors.append(row["filepath"])
            all_extended.append({})
        else:
            all_extended.append(result)

    # Adaugam cei 108 parametri noi la dataframe
    df_extended = pd.DataFrame(all_extended)
    df_out      = pd.concat([df.reset_index(drop=True),
                             df_extended.reset_index(drop=True)], axis=1)

    # Eliminam randurile cu erori (unde toti parametrii extinsi sunt NaN)
    n_before = len(df_out)
    first_new_col = df_extended.columns[0] if len(df_extended.columns) > 0 else None
    if first_new_col:
        df_out = df_out.dropna(subset=[first_new_col])
    n_after = len(df_out)
    if n_before > n_after:
        print(f"\n  Atentie: {n_before - n_after} randuri eliminate (imagini inaccesibile)")

    # Salvam CSV-ul
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    def smart_format(x):
        if isinstance(x, float):
            if x != 0 and abs(x) < 0.01:
                return f"{x:.2e}"
            return f"{round(x, 3)}"
        return x

    df_out = df_out.apply(
        lambda col: col.map(smart_format) if col.dtype == float else col
    )
    df_out.to_csv(output_csv, index=False)

    n_params = len(df_out.columns) - 6
    print(f"\n{'='*60}")
    print(f"  Procesate cu succes : {n_after}")
    print(f"  Erori               : {len(errors)}")
    print(f"  Parametri totali    : {n_params} "
          f"(85 existenti + 108 GLCM extinsi)")
    print(f"  CSV salvat in       : {output_csv}")
    print(f"{'='*60}")

    if errors:
        print(f"\n  Fisiere cu erori:")
        for e in errors[:5]:
            print(f"    - {e}")
        if len(errors) > 5:
            print(f"    ... si alte {len(errors)-5} fisiere")


# ──────────────────────────── ENTRY POINT ───────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Adaugare parametri GLCM extinsi la setul existent"
    )
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT_CSV,
        help="CSV-ul de input (implicit: features_caf.csv)"
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_CSV,
        help="CSV-ul de output (implicit: features_glcm_extended.csv)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_all(input_csv=args.input, output_csv=args.output)