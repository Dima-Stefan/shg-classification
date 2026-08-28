"""
features_R_ratio.py
Construieste o imagine noua "R" pentru fiecare pereche forward/backward
(aceeasi achizitie fizica: probe+sectiune+polarizare), definita per-pixel
ca un raport normalizat si marginit intre 0 si 255:

    R_raw = (F - B) / (F + B)              -- in intervalul [-1, 1]
    R_shifted = R_raw + 1                   -- in [0, 2]
    R_scaled = round(R_shifted * 128)       -- in [0, 256], clampat la 255

Spre deosebire de F/B brut (Sectiunea F/B ratio anterioara), care putea
exploda catre infinit cand B -> 0, acest R e MARGINIT prin constructie
in [-1, 1] indiferent cat de slab e semnalul backward -- rezolva
structural instabilitatea observata (forma de 'palnie' in scatter plot
F/B vs intensitate backward).

Pixelii unde F=0 SI B=0 (fundal complet negru in ambele canale) au
raportul (F-B)/(F+B) nedefinit (0/0) -- setat explicit la R_raw=0
(neutru), corespunzand R_scaled=128 (centrul intervalului).

Odata construita imaginea R (uint8, aceeasi dimensiune ca imaginile
originale), se aplica EXACT acelasi pipeline de extragere folosit
pentru imaginile normale (histogram_features + glcm_features din
features.py), rezultand 64 de parametri comparabili structural cu
setul standard.

Output: Results/CSV/features_R_ratio.csv
    Un rand per pereche (probe, sectiune, polarizare) cu ambele canale
    disponibile.

Utilizare:
    python features_R_ratio.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))


def _find_and_add_core(start: Path) -> bool:
    current = start
    for _ in range(5):
        candidate = current / "core"
        if (candidate / "utils.py").exists():
            sys.path.insert(0, str(candidate))
            return True
        current = current.parent
    return False

_find_and_add_core(Path(__file__).resolve().parent)

from utils import collect_images
from features import load_grayscale, histogram_features, glcm_features

DEFAULT_DATA_DIR   = None
DEFAULT_OUTPUT_CSV = None
try:
    from config import DATA_DIR as CONFIG_DATA_DIR, CSV_DIR
    DEFAULT_DATA_DIR   = CONFIG_DATA_DIR
    DEFAULT_OUTPUT_CSV = CSV_DIR / "features_R_ratio.csv"
except ImportError:
    pass  # fara config.py, --data si --output-csv devin obligatorii


# ──────────────────────────── CONFIGURARE ───────────────────────────────────

CHANNEL_BACK    = "ch00"
CHANNEL_FORWARD = "ch02"

SCALE_FACTOR = 128
MAX_VALUE    = 255  # clamp; formula bruta poate atinge 256 doar la R_raw=1 exact


# ──────────────────────────── CONSTRUCTIE IMAGINE R ──────────────────────────

def build_R_image(img_forward: np.ndarray, img_backward: np.ndarray) -> np.ndarray:
    """
    Construieste imaginea R per-pixel din perechea forward/backward.
    Ambele imagini trebuie sa aiba aceeasi dimensiune.
    """
    F = img_forward.astype(np.float64)
    B = img_backward.astype(np.float64)

    denom = F + B
    with np.errstate(divide="ignore", invalid="ignore"):
        R_raw = np.where(denom > 0, (F - B) / denom, 0.0)  # 0/0 -> 0 (neutru)

    R_shifted = R_raw + 1.0                    # [0, 2]
    R_scaled  = np.round(R_shifted * SCALE_FACTOR)  # [0, 256]
    R_scaled  = np.clip(R_scaled, 0, MAX_VALUE)     # clamp 256 -> 255

    return R_scaled.astype(np.uint8)


def extract_R_features(image_records: list) -> pd.DataFrame:
    """
    Grupeaza inregistrarile dupa (probe, section, polar), construieste
    imaginea R pentru fiecare pereche completa (ch00+ch02), si extrage
    parametrii histogram+GLCM standard pe aceasta imagine noua.
    """
    pairs = {}
    for record in image_records:
        key = (record["probe"], record["section"], record["polar"])
        pairs.setdefault(key, {})[record["channel"]] = record

    rows   = []
    errors = []

    for (probe, section, polar), channels in tqdm(pairs.items(), desc="Extragere R", unit="pereche"):
        if CHANNEL_BACK not in channels or CHANNEL_FORWARD not in channels:
            continue

        back_record    = channels[CHANNEL_BACK]
        forward_record = channels[CHANNEL_FORWARD]

        img_back    = load_grayscale(back_record["filepath"])
        img_forward = load_grayscale(forward_record["filepath"])

        if img_back is None or img_forward is None:
            errors.append((probe, section, polar))
            continue

        if img_back.shape != img_forward.shape:
            errors.append((probe, section, polar))
            continue

        R_img = build_R_image(img_forward, img_back)

        feats = {}
        feats.update(histogram_features(R_img))
        feats.update(glcm_features(R_img))

        row = {
            "probe"   : probe,
            "section" : section,
            "polar"   : polar,
            "is_HE"   : back_record["is_HE"],
        }
        row.update(feats)
        rows.append(row)

    if errors:
        print(f"\n  Atentie: {len(errors)} perechi cu erori (citire sau dimensiuni "
              f"incompatibile), ignorate:")
        for e in errors[:10]:
            print(f"    - {e}")

    return pd.DataFrame(rows)


# ──────────────────────────── PROCESARE PRINCIPALA ──────────────────────────

def process_all(data_dir: Path, output_csv: Path):
    print(f"\n{'='*60}")
    print(f"  Extragere Parametri Imagine R = norm[(F-B)/(F+B)]")
    print(f"  Date    : {data_dir}")
    print(f"  Output  : {output_csv}")
    print(f"{'='*60}\n")

    image_records = collect_images(data_dir)
    if not image_records:
        print("[EROARE] Niciun fisier valid gasit.")
        return

    df = extract_R_features(image_records)

    if df.empty:
        print("[EROARE] Nicio pereche completa (ch00+ch02) gasita.")
        return

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"\n{'='*60}")
    print(f"  Perechi procesate  : {len(df)}")
    print(f"  Parametri per rand : {df.shape[1] - 4} (4 histograma + 60 GLCM)")
    print(f"  CSV salvat in      : {output_csv}")
    print(f"{'='*60}")


# ──────────────────────────── ENTRY POINT ───────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extragere parametri pe imaginea R = norm[(F-B)/(F+B)]"
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_DIR,
                         required=DEFAULT_DATA_DIR is None,
                         help="Folderul cu imagini SHG (Data/)")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV,
                         required=DEFAULT_OUTPUT_CSV is None,
                         help="Calea CSV-ului de output (ex. Results/CSV/features_R_ratio.csv)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_all(args.data, args.output_csv)