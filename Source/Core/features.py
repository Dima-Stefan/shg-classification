"""
features.py
Extragerea parametrilor cantitativi din imaginile SHG.

Parametri calculati:
  Histograma : Mean, StDev, Skew, Kurt
  GLCM       : media pe perechi de directii opuse (0+180, 90+270, 45+225, 135+315)
               la distantele specificate
               primul rand si prima coloana ale GLCM sunt eliminate
               pentru a reduce contributia tranzitiilor background

Numele parametrilor: ex. ASM_0_180_d1, Contrast_90_270_d2 etc.
"""

import numpy as np
from scipy.stats import skew, kurtosis
from skimage import io
from skimage.feature import graycomatrix, graycoprops


# ──────────────────────────── CONFIGURARE ───────────────────────────────────

# Distantele pentru calculul GLCM
GLCM_DISTANCES = [1, 10, 30]

# Perechile de directii opuse (in radiani)
# Media se face pe cele doua directii din fiecare pereche
GLCM_DIRECTION_PAIRS = {
    "0_180"   : [0,            np.pi],           # 0°  + 180°
    "90_270"  : [np.pi / 2,    3 * np.pi / 2],   # 90° + 270°
    "45_225"  : [np.pi / 4,    5 * np.pi / 4],   # 45° + 225°
    "135_315" : [3 * np.pi / 4, 7 * np.pi / 4],  # 135° + 315°
}

# Numarul de nivele de gri
GLCM_LEVELS = 256


# ──────────────────────────── INCARCARE IMAGINE ─────────────────────────────

def load_grayscale(filepath):
    """
    Incarca o imagine .tif si o aduce la grayscale uint8 (0-255).
    Returneaza None daca fisierul nu poate fi citit.
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
        print(f"[features] EROARE la citirea {filepath}: {e}")
        return None


# ──────────────────────────── PARAMETRI HISTOGRAMA ──────────────────────────

def histogram_features(img):
    """
    Calculeaza parametrii de histograma pe imaginea grayscale.
    """
    pixels = img.flatten().astype(np.float64)

    return {
        "Mean" : float(np.mean(pixels)),
        "StDev": float(np.std(pixels, ddof=1)),
        "Skew" : float(skew(pixels)),
        "Kurt" : float(kurtosis(pixels, fisher=True)),
    }


# ──────────────────────────── PARAMETRI GLCM ────────────────────────────────

def compute_glcm_params(glcm_2d):
    """
    Calculeaza parametrii GLCM dintr-o matrice 2D normalizata.
    Primul rand si prima coloana sunt eliminate pentru a reduce
    contributia tranzitiilor background-primul nivel de intensitate.
    """
    # Eliminam primul rand si prima coloana
    glcm_trimmed = glcm_2d[1:, 1:]

    # Renormalizam dupa eliminare
    total = glcm_trimmed.sum()
    if total > 0:
        glcm_trimmed = glcm_trimmed / total
    else:
        return {
            "ASM": 0.0, "Contrast": 0.0,
            "Correlation": 0.0, "IDM": 0.0, "Entropy": 0.0
        }

    n = glcm_trimmed.shape[0]
    i_vals = np.arange(n, dtype=np.float64)

    # ASM
    asm = float(np.sum(glcm_trimmed ** 2))

    # Contrast
    i_grid, j_grid = np.meshgrid(i_vals, i_vals, indexing='ij')
    contrast = float(np.sum((i_grid - j_grid) ** 2 * glcm_trimmed))

    # IDM
    idm = float(np.sum(glcm_trimmed / (1 + (i_grid - j_grid) ** 2)))

    # Correlation (formula Walker)
    px   = float(np.sum(i_vals * glcm_trimmed.sum(axis=1)))
    py   = float(np.sum(i_vals * glcm_trimmed.sum(axis=0)))
    varx = float(np.sum((i_vals - px) ** 2 * glcm_trimmed.sum(axis=1)))
    vary = float(np.sum((i_vals - py) ** 2 * glcm_trimmed.sum(axis=0)))

    if varx * vary > 0:
        correlation = float(
            np.sum((i_grid - px) * (j_grid - py) * glcm_trimmed / (varx * vary))
        )
    else:
        correlation = 0.0

    # Entropy cu log natural (ca in ImageJ)
    nonzero = glcm_trimmed[glcm_trimmed > 0]
    entropy = float(-np.sum(nonzero * np.log(nonzero))) if len(nonzero) > 0 else 0.0

    return {
        "ASM"        : asm,
        "Contrast"   : contrast,
        "Correlation": correlation,
        "IDM"        : idm,
        "Entropy"    : entropy,
    }


def glcm_features_for_pair(img, angles, distance, pair_name):
    """
    Calculeaza parametrii GLCM pentru o pereche de directii opuse
    la o distanta specificata.

    Face media GLCM pe cele doua directii din pereche,
    apoi calculeaza parametrii pe matricea medie.
    """
    # Calculam GLCM pentru ambele directii din pereche
    glcm = graycomatrix(
        img,
        distances=[distance],
        angles=angles,
        levels=GLCM_LEVELS,
        symmetric=False,
        normed=True,
    )
    # glcm shape: (256, 256, 1_distanta, 2_unghiuri)
    # Facem media pe cele 2 directii
    glcm_mean = (glcm[:, :, 0, 0] + glcm[:, :, 0, 1]) / 2

    # Calculam parametrii pe matricea medie
    params = compute_glcm_params(glcm_mean)

    # Adaugam sufixul cu directia si distanta
    suffix = f"_{pair_name}_d{distance}"
    return {f"{k}{suffix}": v for k, v in params.items()}


def glcm_features(img):
    """
    Calculeaza parametrii GLCM pentru toate perechile de directii
    si toate distantele.

    Numele parametrilor: ASM_0_180_d1, Contrast_90_270_d2 etc.
    """
    feats = {}
    for pair_name, angles in GLCM_DIRECTION_PAIRS.items():
        for dist in GLCM_DISTANCES:
            feats.update(glcm_features_for_pair(img, angles, dist, pair_name))
    return feats


# ──────────────────────────── EXTRAGERE COMPLETA ────────────────────────────

def extract_features(filepath):
    """
    Incarca o imagine si calculeaza toti parametrii.
    Returneaza None daca imaginea nu poate fi procesata.
    """
    img = load_grayscale(filepath)
    if img is None:
        return None

    features = {}
    features.update(histogram_features(img))
    features.update(glcm_features(img))
    return features