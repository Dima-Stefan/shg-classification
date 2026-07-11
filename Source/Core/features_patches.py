"""
patch_features.py
Taie imaginile SHG in patch-uri de 32x32 pixeli, filtreaza patch-urile
cu semnal insuficient si extrage parametrii histogram + GLCM pentru
fiecare patch valid.

Rezultatul e salvat in Results/features_patches.csv — separat de
features.csv (imaginile mari) care raman intacte.

Utilizare:
    python patch_features.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Importam modulele existente
sys.path.insert(0, str(Path(__file__).parent))
from utils    import collect_images
from features import load_grayscale, histogram_features, glcm_features

# ──────────────────────────── CONFIGURARE ───────────────────────────────────

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
from config import DATA_DIR, CSV_FEATURES_PATCHES as OUTPUT_CSV

# Dimensiunea patch-ului(mai intai a fost 32, apoi 64, apoi 128 si dupa 256)
PATCH_SIZE = 256

# Pragul pentru media pixelilor (patch-urile sub acest prag sunt eliminate, era 2.76 pt 128x128 si 4.09 pt 256x256)
MEAN_THRESHOLD = 2


# ──────────────────────────── TAIERE IN PATCH-URI ───────────────────────────

def extract_patches(img: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
    """
    Imparte imaginea 512x512 in exact 4 patch-uri de 256x256
    corespunzatoare celor 4 colturi:
        - sus-stanga  : [0:256,   0:256]
        - sus-dreapta : [0:256,   256:512]
        - jos-stanga  : [256:512, 0:256]
        - jos-dreapta : [256:512, 256:512]
    """
    h, w = img.shape
    half_h = h // 2
    half_w = w // 2

    corners = [
        (0,      0,      img[0:half_h,      0:half_w]),       # sus-stanga
        (0,      half_w, img[0:half_h,      half_w:w]),       # sus-dreapta
        (half_h, 0,      img[half_h:h,      0:half_w]),       # jos-stanga
        (half_h, half_w, img[half_h:h,      half_w:w]),       # jos-dreapta
    ]

    # Filtram patch-urile cu semnal insuficient
    patches = [
        (row, col, patch)
        for row, col, patch in corners
        if patch.mean() >= MEAN_THRESHOLD
    ]

    return patches


# ──────────────────────────── PROCESARE PRINCIPALA ──────────────────────────

def process_all():
    print(f"\n{'='*60}")
    print(f"  SHG Patch Feature Extraction (32x32)")
    print(f"{'='*60}")
    print(f"  Date      : {DATA_DIR}")
    print(f"  Output    : {OUTPUT_CSV}")
    print(f"  Patch size: {PATCH_SIZE}x{PATCH_SIZE} px")
    print(f"  Prag medie: {MEAN_THRESHOLD}")
    print(f"{'='*60}\n")

    # ── 1. Colectam imaginile valide (la fel ca main.py) ────────────────────
    image_records = collect_images(DATA_DIR)
    if not image_records:
        print("[EROARE] Niciun fisier valid gasit.")
        return

    # ── 2. Procesam fiecare imagine ─────────────────────────────────────────
    rows          = []
    errors        = []
    total_patches = 0
    kept_patches  = 0

    for record in tqdm(image_records, desc="Procesare imagini", unit="img"):
        img = load_grayscale(record["filepath"])
        if img is None:
            errors.append(record["filepath"])
            continue

        patches = extract_patches(img)
        total_patches += (img.shape[0] // PATCH_SIZE) * (img.shape[1] // PATCH_SIZE)
        kept_patches  += len(patches)

        for row_idx, col_idx, patch in patches:
            # Calculam parametrii pentru patch
            feats = {}
            feats.update(histogram_features(patch))
            feats.update(glcm_features(patch))

            # Construim randul cu metadate + parametri
            row = {
            # ── Metadate imagine parinte ───────────────────────────────
            "probe"     : record["probe"],
            "section"   : record["section"],
            "polar"     : record["polar"],
            "channel"   : record["channel"],
            "is_HE"     : record["is_HE"],
            # ── Pozitia patch-ului in imaginea parinte ─────────────────
            "patch_row" : row_idx,
            "patch_col" : col_idx,
            # ── Parametri histogram ────────────────────────────────────
            "Mean"      : feats["Mean"],
            "StDev"     : feats["StDev"],
            "Skew"      : feats["Skew"],
            "Kurt"      : feats["Kurt"],
        }

            # Adaugam toti parametrii GLCM dinamic
            for key, value in feats.items():
                if key not in ["Mean", "StDev", "Skew", "Kurt"]:
                    row[key] = value

                rows.append(row)

    # ── 3. Salvare CSV ──────────────────────────────────────────────────────
    if not rows:
        print("[EROARE] Niciun patch valid gasit.")
        return

    df = pd.DataFrame(rows)

    # Sortam la fel ca in main.py
    df["polar_num"] = df["polar"].str.extract(r"(\d+)").astype(float).fillna(999)
    df = df.sort_values(
        ["probe", "section", "channel", "polar_num", "patch_row", "patch_col"]
    ).reset_index(drop=True)
    df = df.drop(columns=["polar_num"])

    # Formatare inteligenta: 2 zecimale, stiintific pentru valori mici
    def smart_format(x):
        if isinstance(x, float):
            if x != 0 and abs(x) < 0.01:
                return f"{x:.2e}"
            return f"{round(x, 2)}"
        return x

    df = df.apply(lambda col: col.map(smart_format) if col.dtype == float else col)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    # ── 4. Sumar ─────────────────────────────────────────────────────────────
    pct_kept = kept_patches / total_patches * 100 if total_patches > 0 else 0

    print(f"\n{'='*60}")
    print(f"  Imagini procesate    : {len(image_records) - len(errors)}")
    print(f"  Erori                : {len(errors)}")
    print(f"  Total patch-uri      : {total_patches}")
    print(f"  Patch-uri pastrate   : {kept_patches} ({pct_kept:.1f}%)")
    print(f"  Patch-uri eliminate  : {total_patches - kept_patches} (medie < {MEAN_THRESHOLD})")
    print(f"  Randuri in CSV       : {len(df)}")
    print(f"  CSV salvat in        : {OUTPUT_CSV}")
    print(f"{'='*60}")

    if errors:
        print(f"\n  Fisiere cu erori:")
        for e in errors:
            print(f"    - {e}")


# ──────────────────────────── ENTRY POINT ───────────────────────────────────

if __name__ == "__main__":
    process_all()