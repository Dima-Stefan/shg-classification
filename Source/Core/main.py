"""
main.py
Script principal — parcurge toate folderele din Data/, extrage parametrii
din fiecare imagine SHG valida si salveaza rezultatele in Results/features.csv.

Utilizare:
    python main.py

    Optional, puteti specifica alt folder de date:
    python main.py --data ../Data --output ../Results/features.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm  # bara de progres — instaleaza cu: pip install tqdm

# Importam modulele noastre
sys.path.insert(0, str(Path(__file__).parent))
from utils    import collect_images
from features import extract_features


# ──────────────────────────── CALE IMPLICITA ────────────────────────────────

# Ajusteaza aceste cai daca structura ta de foldere difera
DEFAULT_DATA_DIR   = Path(__file__).parent.parent / "Data"
DEFAULT_OUTPUT_CSV = Path(__file__).parent.parent / "Results" / "features.csv"


# ──────────────────────────── PROCESARE ─────────────────────────────────────

def process_all(data_dir: Path, output_csv: Path) -> None:
    """
    Functia principala:
      1. Colecteaza toate imaginile valide din data_dir
      2. Extrage parametrii pentru fiecare imagine
      3. Salveaza CSV-ul in output_csv
    """

    # ── 1. Colectare fisiere ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  SHG Feature Extraction")
    print(f"{'='*60}")
    print(f"  Date    : {data_dir}")
    print(f"  Output  : {output_csv}")
    print(f"{'='*60}\n")

    image_records = collect_images(data_dir)

    if not image_records:
        print("[main] Niciun fisier valid gasit. Verifica calea catre Data/.")
        return

    # ── 2. Extragere parametri ──────────────────────────────────────────────
    rows = []
    errors = []

    for record in tqdm(image_records, desc="Procesare imagini", unit="img"):
        feats = extract_features(record["filepath"])

        if feats is None:
            errors.append(record["filepath"])
            continue

        # Combinam metadatele cu parametrii calculati
        row = {
            # ── Metadata ──────────────────────────────────────────────────
            "probe"   : record["probe"],
            "section" : record["section"],
            "polar"   : record["polar"],
            "channel" : record["channel"],
            "is_HE"   : record["is_HE"],
            "filepath": record["filepath"],
            # ── Parametri histograma ───────────────────────────────────────
            "Mean" : feats["Mean"],
            "StDev": feats["StDev"],
            "Skew" : feats["Skew"],
            "Kurt" : feats["Kurt"],
        }

        # Adaugam toti parametrii GLCM dinamic
        for key, value in feats.items():
            if key not in ["Mean", "StDev", "Skew", "Kurt"]:
                row[key] = value

        rows.append(row)

    # ── 3. Salvare CSV ──────────────────────────────────────────────────────
    if not rows:
        print("[main] Nicio imagine procesata cu succes.")
        return

    df = pd.DataFrame(rows)

    # Sortam pentru claritate
    df["polar_num"] = df["polar"].str.extract(r"(\d+)").astype(float).fillna(999)
    df = df.sort_values(["probe", "section", "channel", "polar_num"]).reset_index(drop=True)
    df = df.drop(columns=["polar_num"])

    # Cream folderul Results/ daca nu exista
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    # Formatare inteligenta: 2 zecimale normal, stiintific pentru valori mici
    def smart_format(x):
        if isinstance(x, float):
            if x != 0 and abs(x) < 0.01:
                return f"{x:.2e}"   # ex: 8.37e-04
            return f"{round(x, 3)}" # ex: 6.152
        return x

    df = df.apply(lambda col: col.map(smart_format) if col.dtype == float else col)
    df.to_csv(output_csv, index=False)

    # ── 4. Sumar ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Procesate cu succes : {len(rows)}")
    print(f"  Erori               : {len(errors)}")
    print(f"  CSV salvat in       : {output_csv}")
    print(f"{'='*60}")
    print(f"\n  Probe unice  : {df['probe'].nunique()}")
    print(f"  ch00 (back)  : {(df['channel']=='ch00').sum()} imagini")
    print(f"  ch02 (forward): {(df['channel']=='ch02').sum()} imagini")
    print(f"  HE colorate  : {df['is_HE'].sum()} imagini")
    print(f"  Necolorate   : {(~df['is_HE']).sum()} imagini")

    if errors:
        print(f"\n  Fisiere cu erori:")
        for e in errors:
            print(f"    - {e}")


# ──────────────────────────── ENTRY POINT ───────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extragere parametri SHG (histograma + GLCM)"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Folderul cu imagini (implicit: {DEFAULT_DATA_DIR})"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help=f"Fisierul CSV de output (implicit: {DEFAULT_OUTPUT_CSV})"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_all(data_dir=args.data, output_csv=args.output)
