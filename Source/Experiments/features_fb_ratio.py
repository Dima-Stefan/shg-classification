"""
features_fb_ratio.py
Calculul raportului Forward/Backward (F/B) al semnalului SHG -- un
parametru standard in literatura de microscopie SHG, care cuantifica
gradul de organizare/maturare a fibrelor de colagen. Colagenul dens,
bine organizat imprastie SHG preferential inainte (forward), in timp
ce colagenul dezorganizat/subtire imprastie mai simetric in ambele
directii -- deci F/B mare tinde sa indice colagen matur/organizat,
F/B mic colagen dezorganizat.

F/B = intensitate totala forward (ch02) / intensitate totala backward (ch00)

Calculat DOAR pentru perechile de imagini care provin din aceeasi
achizitie fizica (acelasi probe, aceeasi sectiune, aceeasi polarizare),
identic cu potrivirea din classify_svm_combined_channels.py.

Intensitatea totala per imagine = suma tuturor pixelilor (echivalent
cu Mean * numar_pixeli, dar calculat direct pe imagine pentru precizie).

Output: Results/CSV/features_fb_ratio.csv
    Randuri: cate o pereche (probe, sectiune, polarizare) cu ambele
    canale disponibile.
    Coloane: metadate + FB_ratio.

Utilizare:
    python features_fb_ratio.py
"""

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
from features import load_grayscale

try:
    from config import DATA_DIR, CSV_DIR
    OUTPUT_CSV = CSV_DIR / "features_fb_ratio.csv"
except ImportError:
    DATA_DIR   = Path(r"C:\Users\dimas\Downloads\projects\venv\Data")
    OUTPUT_CSV = Path(r"C:\Users\dimas\Downloads\projects\venv\Results\CSV\features_fb_ratio.csv")


# ──────────────────────────── CONFIGURARE ───────────────────────────────────

CHANNEL_BACK    = "ch00"
CHANNEL_FORWARD = "ch02"


# ──────────────────────────── CALCUL F/B ────────────────────────────────────

def total_intensity(img: np.ndarray) -> float:
    """Suma tuturor pixelilor unei imagini grayscale (proportionala cu
    energia totala a semnalului SHG captat)."""
    return float(np.sum(img.astype(np.float64)))


def compute_fb_ratios(image_records: list) -> pd.DataFrame:
    """
    Grupeaza inregistrarile dupa (probe, section, polar) si calculeaza
    F/B pentru fiecare pereche unde ambele canale (ch00 si ch02) sunt
    disponibile.
    """
    # Organizam pe cheie (probe, section, polar) -> {channel: filepath}
    pairs = {}
    for record in image_records:
        key = (record["probe"], record["section"], record["polar"])
        pairs.setdefault(key, {})[record["channel"]] = record

    rows = []
    errors = []

    for (probe, section, polar), channels in tqdm(pairs.items(), desc="Calculare F/B", unit="pereche"):
        if CHANNEL_BACK not in channels or CHANNEL_FORWARD not in channels:
            continue  # pereche incompleta, sarim peste

        back_record    = channels[CHANNEL_BACK]
        forward_record = channels[CHANNEL_FORWARD]

        img_back    = load_grayscale(back_record["filepath"])
        img_forward = load_grayscale(forward_record["filepath"])

        if img_back is None or img_forward is None:
            errors.append((probe, section, polar))
            continue

        intensity_back    = total_intensity(img_back)
        intensity_forward = total_intensity(img_forward)

        if intensity_back == 0:
            fb_ratio = np.nan  # evitam divide-by-zero; randul va fi eliminat ulterior
        else:
            fb_ratio = intensity_forward / intensity_back

        rows.append({
            "probe"             : probe,
            "section"           : section,
            "polar"             : polar,
            "is_HE"             : back_record["is_HE"],
            "intensity_back"    : intensity_back,
            "intensity_forward" : intensity_forward,
            "FB_ratio"          : fb_ratio,
        })

    if errors:
        print(f"\n  Atentie: {len(errors)} perechi cu erori la citire, ignorate:")
        for e in errors[:10]:
            print(f"    - {e}")

    return pd.DataFrame(rows)


# ──────────────────────────── PROCESARE PRINCIPALA ──────────────────────────

def process_all(data_dir: Path, output_csv: Path):
    print(f"\n{'='*60}")
    print(f"  Calcul Raport Forward/Backward (F/B)")
    print(f"  Date    : {data_dir}")
    print(f"  Output  : {output_csv}")
    print(f"{'='*60}\n")

    image_records = collect_images(data_dir)
    if not image_records:
        print("[EROARE] Niciun fisier valid gasit.")
        return

    df = compute_fb_ratios(image_records)

    if df.empty:
        print("[EROARE] Nicio pereche completa (ch00+ch02) gasita.")
        return

    n_before = len(df)
    df = df.dropna(subset=["FB_ratio"])
    if len(df) < n_before:
        print(f"  Atentie: {n_before - len(df)} randuri eliminate (FB_ratio invalid, "
              f"probabil intensitate backward = 0)")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"\n{'='*60}")
    print(f"  Perechi complete procesate : {len(df)}")
    print(f"  FB_ratio mediu             : {df['FB_ratio'].mean():.3f}")
    print(f"  FB_ratio min / max         : {df['FB_ratio'].min():.3f} / {df['FB_ratio'].max():.3f}")
    print(f"  CSV salvat in              : {output_csv}")
    print(f"{'='*60}")


if __name__ == "__main__":
    process_all(DATA_DIR, OUTPUT_CSV)