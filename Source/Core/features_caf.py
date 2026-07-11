"""
features_caf.py
Implementarea CAF (Collagen Area Fraction) — fractia de pixeli cu
semnal SHG prezent (intensitate > prag) din imaginea totala.

CAF = numar pixeli cu intensitate > CAF_THRESHOLD / numar total pixeli

Este cel mai simplu parametru posibil: un singur numar per imagine,
fara calcule de matrice sau directii. Masoara cat de mult din imaginea
SHG este ocupata de colagen vizibil.

Interpretare biologica:
  - CAF mare -> tesut cu colagen dens, bine reprezentat in imagine
  - CAF mic  -> imagine cu mult fundal negru (putine fibre vizibile)

Prag configurabil (CAF_THRESHOLD):
  - 0 (implicit, conform articolului de referinta): orice pixel nenul
  - Valori mai mari (ex. 5, 10): exclud zgomotul de fond din aparatura

Utilizare (importat de main_caf.py):
    from features_caf import caf_features, extract_all_features
"""

import numpy as np
from skimage import io

# ──────────────────────────── CONFIGURARE ───────────────────────────────────

# Pragul de intensitate peste care un pixel e considerat semnal SHG.
# 0 = orice pixel nenul (conform articolului de referinta).
# Poate fi suprascris la import: features_caf.CAF_THRESHOLD = 10
CAF_THRESHOLD = 0


# ──────────────────────────── INCARCARE IMAGINE ─────────────────────────────

def load_grayscale(filepath):
    """
    Incarca o imagine .tif si o aduce la grayscale uint8 (0-255).
    Identic cu features.py — returneaza None daca fisierul nu poate
    fi citit.
    """
    try:
        img = io.imread(str(filepath))

        if img.ndim == 3:
            img = img[:, :, 0]

        if img.dtype != np.uint8:
            img = img.astype(np.float64)
            min_val, max_val = img.min(), img.max()
            if max_val > min_val:
                img = (img - min_val) / (max_val - min_val) * 255
            img = img.astype(np.uint8)

        return img

    except Exception as e:
        print(f"[features_caf] EROARE la citirea {filepath}: {e}")
        return None


# ──────────────────────────── CAF ───────────────────────────────────────────

def caf_features(img, threshold=None):
    """
    Calculeaza CAF (Collagen Area Fraction) pentru o imagine grayscale.

    Returneaza un dictionar cu un singur parametru:
        CAF: fractia de pixeli cu intensitate > threshold (float, 0-1)

    Parametri:
        img       : imagine grayscale uint8 (numpy array)
        threshold : pragul de intensitate (implicit CAF_THRESHOLD global)
    """
    if threshold is None:
        threshold = CAF_THRESHOLD

    total_pixels   = img.size
    signal_pixels  = int(np.sum(img > threshold))

    caf = signal_pixels / total_pixels if total_pixels > 0 else 0.0

    return {"CAF": float(caf)}


# ──────────────────────────── EXTRAGERE ─────────────────────────────────────

def extract_caf_features(filepath, threshold=None):
    """
    Incarca o imagine si calculeaza CAF.
    Returneaza None daca imaginea nu poate fi procesata.
    """
    img = load_grayscale(filepath)
    if img is None:
        return None

    return caf_features(img, threshold)