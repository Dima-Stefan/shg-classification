"""
main_glrlm.py
Identic ca structura cu main.py, dar combina cei 64 de parametri
standard (histograma + GLCM, din features.py) cu cei 20 de parametri
noi GLRLM (din features_glrlm.py), rezultand 84 de parametri in total.

Salveaza intr-un CSV separat: Results/features_glrlm.csv

Utilizare:
    python main_glrlm.py
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils         import collect_images
from features      import extract_features as extract_standard_features
from features_glrlm import extract_glrlm_features


# ──────────────────────────── CALE IMPLICITA ────────────────────────────────

DEFAULT_DATA_DIR   = Path(__file__).parent.parent / "Data"
DEFAULT_OUTPUT_CSV = Path(__file__).parent.parent / "Results" / "features_glrlm.csv"


# ──────────────────────────── PROCESARE ─────────────────────────────────────

def process_all(data_dir: Path, output_csv: Path) -> None:
    print(f"\n{'='*60}")
    print(f"  SHG Feature Extraction — Histograma + GLCM + GLRLM")
    print(f"{'='*60}")
    print(f"  Date    : {data_dir}")
    print(f"  Output  : {output_csv}")
    print(f"{'='*60}\n")

    image_records = collect_images(data_dir)

    if not image_records:
        print("[main_glrlm] Niciun fisier valid gasit. Verifica calea catre Data/.")
        return

    rows = []
    errors = []

    for record in tqdm(image_records, desc="Procesare imagini", unit="img"):
        feats_standard = extract_standard_features(record["filepath"])
        feats_glrlm    = extract_glrlm_features(record["filepath"])

        if feats_standard is None or feats_glrlm is None:
            errors.append(record["filepath"])
            continue

        row = {
            "probe"   : record["probe"],
            "section" : record["section"],
            "polar"   : record["polar"],
            "channel" : record["channel"],
            "is_HE"   : record["is_HE"],
            "filepath": record["filepath"],
            "Mean" : feats_standard["Mean"],
            "StDev": feats_standard["StDev"],
            "Skew" : feats_standard["Skew"],
            "Kurt" : feats_standard["Kurt"],
        }

        # Parametrii GLCM standard (60)
        for key, value in feats_standard.items():
            if key not in ["Mean", "StDev", "Skew", "Kurt"]:
                row[key] = value

        # Parametrii GLRLM noi (20)
        for key, value in feats_glrlm.items():
            row[key] = value

        rows.append(row)

    if not rows:
        print("[main_glrlm] Nicio imagine procesata cu succes.")
        return

    df = pd.DataFrame(rows)

    df["polar_num"] = df["polar"].str.extract(r"(\d+)").astype(float).fillna(999)
    df = df.sort_values(["probe", "section", "channel", "polar_num"]).reset_index(drop=True)
    df = df.drop(columns=["polar_num"])

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    def smart_format(x):
        if isinstance(x, float):
            if x != 0 and abs(x) < 0.01:
                return f"{x:.2e}"
            return f"{round(x, 3)}"
        return x

    df = df.apply(lambda col: col.map(smart_format) if col.dtype == float else col)
    df.to_csv(output_csv, index=False)

    print(f"\n{'='*60}")
    print(f"  Procesate cu succes : {len(rows)}")
    print(f"  Erori               : {len(errors)}")
    print(f"  CSV salvat in       : {output_csv}")
    print(f"  Coloane parametri   : {len(df.columns) - 6} "
          f"(64 standard + 20 GLRLM)")
    print(f"{'='*60}")

    if errors:
        print(f"\n  Fisiere cu erori:")
        for e in errors:
            print(f"    - {e}")


# ──────────────────────────── ENTRY POINT ───────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extragere parametri SHG: histograma + GLCM + GLRLM"
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