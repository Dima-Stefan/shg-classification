"""
main_caf.py
Combina cei 84 de parametri existenti (histograma + GLCM + GLRLM,
din features_glrlm.csv) cu parametrul nou CAF (Collagen Area Fraction),
rezultand 85 de parametri in total.

Salveaza in Results/features_caf.csv.

De ce porneste de la features_glrlm.csv si nu de la imagini:
  - Evita recalcularea celor 84 de parametri deja calculati
  - CAF se calculeaza rapid direct din imagini (un singur numar per
    imagine), deci costul e neglijabil
  - features_glrlm.csv contine deja calea catre fiecare imagine
    in coloana 'filepath', folosita pentru a incarca imaginea si
    a calcula CAF

Pragul CAF implicit: 0 (orice pixel nenul = semnal SHG prezent).
Poate fi schimbat prin argumentul --caf-threshold.

Utilizare:
    python main_caf.py
    python main_caf.py --caf-threshold 5
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from features_caf import extract_caf_features


# ──────────────────────────── CALE IMPLICITA ────────────────────────────────

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
from config import CSV_FEATURES_GLRLM as DEFAULT_INPUT_CSV, CSV_FEATURES_CAF as DEFAULT_OUTPUT_CSV


# ──────────────────────────── PROCESARE ─────────────────────────────────────

def process_all(input_csv: Path, output_csv: Path, caf_threshold: int) -> None:
    print(f"\n{'='*60}")
    print(f"  SHG Feature Extraction — Adaugare CAF")
    print(f"{'='*60}")
    print(f"  Input  : {input_csv}")
    print(f"  Output : {output_csv}")
    print(f"  Prag CAF: intensitate > {caf_threshold}")
    print(f"{'='*60}\n")

    if not input_csv.exists():
        print(f"[EROARE] Nu gasesc CSV-ul la: {input_csv}")
        sys.exit(1)

    df = pd.read_csv(input_csv)
    print(f"  Randuri incarcate din features_glrlm.csv: {len(df)}")

    if "filepath" not in df.columns:
        print(f"[EROARE] CSV-ul nu contine coloana 'filepath'. "
              f"Verifica ca folosesti features_glrlm.csv generat de main_glrlm.py.")
        sys.exit(1)

    # Calculam CAF pentru fiecare imagine
    caf_values = []
    errors     = []

    for _, row in tqdm(df.iterrows(), desc="Calculare CAF", total=len(df), unit="img"):
        result = extract_caf_features(row["filepath"], threshold=caf_threshold)
        if result is None:
            errors.append(row["filepath"])
            caf_values.append(None)
        else:
            caf_values.append(result["CAF"])

    # Adaugam coloana CAF in dataframe
    df["CAF"] = caf_values

    # Eliminam randurile unde CAF nu a putut fi calculat
    n_before = len(df)
    df = df.dropna(subset=["CAF"])
    n_after  = len(df)
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

    df_out = df.copy()
    # Formatam doar coloanele float, nu si CAF care e deja float dar intre 0-1
    for col in df_out.select_dtypes(include="float64").columns:
        df_out[col] = df_out[col].apply(smart_format)

    df_out.to_csv(output_csv, index=False)

    # Sumar
    print(f"\n{'='*60}")
    print(f"  Procesate cu succes : {n_after}")
    print(f"  Erori               : {len(errors)}")
    print(f"  Nr. parametri       : {len(df.columns) - 6} "
          f"(84 existenti + 1 CAF)")
    print(f"  CAF medie           : {df['CAF'].mean():.3f}")
    print(f"  CAF min / max       : {df['CAF'].min():.3f} / {df['CAF'].max():.3f}")
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
        description="Adaugare CAF la setul de 84 parametri existenti"
    )
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT_CSV,
        help=f"CSV-ul de input (implicit: features_glrlm.csv)"
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_CSV,
        help=f"CSV-ul de output (implicit: features_caf.csv)"
    )
    parser.add_argument(
        "--caf-threshold", type=int, default=0,
        help="Pragul de intensitate pentru CAF (implicit: 0)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_all(
        input_csv=args.input,
        output_csv=args.output,
        caf_threshold=args.caf_threshold,
    )