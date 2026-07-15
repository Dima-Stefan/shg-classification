"""
main_multi_distance.py
Extrage parametrii histograma + GLCM pentru TOATE cele 7 distante
individuale {1, 5, 10, 15, 20, 25, 30}, pe imaginile 512x512.

Scopul: sa avem un singur CSV care contine GLCM calculat separat la
fiecare distanta, astfel incat sa putem testa ulterior ORICE combinatie
de 3 distante (test_distance_combinations.py) fara sa recalculam GLCM
de 35 de ori -- doar selectam coloanele corespunzatoare din acest CSV.

Foloseste exact aceeasi logica de calcul GLCM ca features.py (trim
primul rand/coloana, renormalizare, medie pe perechi de directii opuse),
doar ca extinsa la 7 distante in loc de 3.

Output: Results/CSV/features_multi_distance.csv
    4 histograma + 5 parametri GLCM x 4 directii x 7 distante = 144 parametri

Utilizare:
    python main_multi_distance.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.feature import graycomatrix

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

from utils import collect_images
from features import load_grayscale, histogram_features, compute_glcm_params

try:
    from config import DATA_DIR, CSV_DIR
    OUTPUT_CSV = CSV_DIR / "features_multi_distance.csv"
except ImportError:
    # Fallback daca structura config.py nu e pe path -- ajusteaza manual
    DATA_DIR   = Path(r"C:\Users\dimas\Downloads\projects\venv\Data")
    OUTPUT_CSV = Path(r"C:\Users\dimas\Downloads\projects\venv\Results\CSV\features_multi_distance.csv")


# ──────────────────────────── CONFIGURARE ───────────────────────────────────

# Toate distantele individuale testate de profesor
ALL_DISTANCES = [1, 5, 10, 15, 20, 25, 30]

GLCM_DIRECTION_PAIRS = {
    "0_180"   : [0,             np.pi],
    "90_270"  : [np.pi / 2,     3 * np.pi / 2],
    "45_225"  : [np.pi / 4,     5 * np.pi / 4],
    "135_315" : [3 * np.pi / 4, 7 * np.pi / 4],
}

GLCM_LEVELS = 256


# ──────────────────────────── CALCUL GLCM MULTI-DISTANTA ────────────────────

def glcm_features_for_pair(img, angles, distance, pair_name):
    """Identic metodologic cu features.py, generalizat pentru orice distanta."""
    glcm = graycomatrix(
        img,
        distances=[distance],
        angles=angles,
        levels=GLCM_LEVELS,
        symmetric=False,
        normed=True,
    )
    glcm_mean = (glcm[:, :, 0, 0] + glcm[:, :, 0, 1]) / 2
    params = compute_glcm_params(glcm_mean)
    suffix = f"_{pair_name}_d{distance}"
    return {f"{k}{suffix}": v for k, v in params.items()}


def glcm_features_all_distances(img):
    """Calculeaza GLCM pentru toate cele 4 perechi x 7 distante = 140 parametri."""
    feats = {}
    for pair_name, angles in GLCM_DIRECTION_PAIRS.items():
        for dist in ALL_DISTANCES:
            feats.update(glcm_features_for_pair(img, angles, dist, pair_name))
    return feats


def extract_all_features(filepath):
    img = load_grayscale(filepath)
    if img is None:
        return None
    feats = {}
    feats.update(histogram_features(img))
    feats.update(glcm_features_all_distances(img))
    return feats


# ──────────────────────────── PROCESARE PRINCIPALA ──────────────────────────

def process_all(data_dir: Path, output_csv: Path):
    print(f"\n{'='*60}")
    print(f"  SHG Multi-Distance GLCM Extraction")
    print(f"  Distante: {ALL_DISTANCES}")
    print(f"  Date    : {data_dir}")
    print(f"  Output  : {output_csv}")
    print(f"{'='*60}\n")

    image_records = collect_images(data_dir)
    if not image_records:
        print("[EROARE] Niciun fisier valid gasit.")
        return

    rows, errors = [], []

    for record in tqdm(image_records, desc="Procesare imagini", unit="img"):
        feats = extract_all_features(record["filepath"])
        if feats is None:
            errors.append(record["filepath"])
            continue

        row = {
            "probe"   : record["probe"],
            "section" : record["section"],
            "polar"   : record["polar"],
            "channel" : record["channel"],
            "is_HE"   : record["is_HE"],
            "filepath": record["filepath"],
        }
        row.update(feats)
        rows.append(row)

    if not rows:
        print("[EROARE] Nicio imagine procesata cu succes.")
        return

    df = pd.DataFrame(rows)
    df["polar_num"] = df["polar"].str.extract(r"(\d+)").astype(float).fillna(999)
    df = df.sort_values(["probe", "section", "channel", "polar_num"]).reset_index(drop=True)
    df = df.drop(columns=["polar_num"])

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"\n{'='*60}")
    print(f"  Procesate cu succes : {len(rows)}")
    print(f"  Erori               : {len(errors)}")
    print(f"  Parametri totali    : {df.shape[1] - 6} (4 hist + 140 GLCM)")
    print(f"  CSV salvat in       : {output_csv}")
    print(f"{'='*60}")

    if errors:
        print(f"\n  Fisiere cu erori:")
        for e in errors:
            print(f"    - {e}")


if __name__ == "__main__":
    process_all(DATA_DIR, OUTPUT_CSV)