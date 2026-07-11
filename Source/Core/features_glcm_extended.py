"""
features_glcm_extended.py
Extensie a features.py care adauga 9 parametri GLCM noi (Grup 1 + Grup 2
din articolul de referinta despre carcinomul tiroidian papilar):

Grup 1 — bazati direct pe GLCM:
    Autocorrelation    : suma(i*j*P(i,j))
    Dissimilarity      : suma(|i-j| * P(i,j))
    Homogeneity        : suma(P(i,j) / (1 + |i-j|))  -- diferit de IDM
    MaxProbability     : max(P(i,j))
    SumOfSquares       : suma((i-mu)^2 * P(i,j))  -- varianta globala

Grup 2 — bazati pe distributiile sumei/diferentei:
    SumAverage         : suma(k * p_{x+y}[k])
    SumEntropy         : -suma(p_{x+y}[k] * ln(p_{x+y}[k]))
    DifferenceVariance : varianta distributiei |i-j|
    DifferenceEntropy  : -suma(p_{x-y}[k] * ln(p_{x-y}[k]))

Calculati pentru aceleasi 4 perechi de directii si 3 distante ca si
parametrii GLCM standard din features.py, rezultand 9*4*3 = 108
parametri noi.

Nota despre Homogeneity vs IDM:
    IDM (din features.py)  = suma(P(i,j) / (1 + (i-j)^2))
    Homogeneity (acest fis) = suma(P(i,j) / (1 + |i-j|))
    Formula cu |i-j| descreaste mai lent decat cea cu (i-j)^2 --
    mai putin sensibila la diferente mari.

Utilizare (importat de main_glcm_extended.py):
    from features_glcm_extended import glcm_extended_features
"""

import numpy as np
from skimage import io
from skimage.feature import graycomatrix


# ──────────────────────────── CONFIGURARE ───────────────────────────────────

GLCM_DISTANCES = [1, 10, 30]

GLCM_DIRECTION_PAIRS = {
    "0_180"   : [0,             np.pi],
    "90_270"  : [np.pi / 2,    3 * np.pi / 2],
    "45_225"  : [np.pi / 4,    5 * np.pi / 4],
    "135_315" : [3 * np.pi / 4, 7 * np.pi / 4],
}

GLCM_LEVELS = 256


# ──────────────────────────── INCARCARE IMAGINE ─────────────────────────────

def load_grayscale(filepath):
    """Identic cu features.py."""
    try:
        img = io.imread(str(filepath))
        if img.ndim == 3:
            img = img[:, :, 0]
        if img.dtype != np.uint8:
            img = img.astype(np.float64)
            mn, mx = img.min(), img.max()
            if mx > mn:
                img = (img - mn) / (mx - mn) * 255
            img = img.astype(np.uint8)
        return img
    except Exception as e:
        print(f"[features_glcm_extended] EROARE la citirea {filepath}: {e}")
        return None


# ──────────────────────────── PARAMETRI GLCM EXTINSI ────────────────────────

def compute_glcm_extended_params(glcm_2d):
    """
    Calculeaza cei 9 parametri GLCM extinsi dintr-o matrice 2D.

    Aplica aceleasi conventii ca features.py:
      - elimina primul rand si prima coloana (trim background)
      - renormalizeaza dupa eliminare
      - calculeaza parametrii pe matricea trimmed/renormalizata

    Toate calculele sunt vectorizate (fara bucle Python) pentru
    performanta pe matrice de 255x255.
    """
    # Trim + renormalizare (identic cu features.py)
    glcm = glcm_2d[1:, 1:]
    total = glcm.sum()
    if total == 0:
        return {k: 0.0 for k in [
            "Autocorrelation", "Dissimilarity", "Homogeneity",
            "MaxProbability", "SumOfSquares",
            "SumAverage", "SumEntropy",
            "DifferenceVariance", "DifferenceEntropy",
        ]}
    glcm = glcm / total

    n = glcm.shape[0]
    i_vals          = np.arange(n, dtype=np.float64)
    i_grid, j_grid  = np.meshgrid(i_vals, i_vals, indexing='ij')
    diff_grid       = np.abs(i_grid - j_grid)

    # ── GRUP 1 ────────────────────────────────────────────────────────────

    # Autocorrelation
    autocorr = float(np.sum(i_grid * j_grid * glcm))

    # Dissimilarity
    dissimilarity = float(np.sum(diff_grid * glcm))

    # Homogeneity (|i-j| la numitor, nu (i-j)^2 ca IDM)
    homogeneity = float(np.sum(glcm / (1.0 + diff_grid)))

    # Maximum Probability
    max_prob = float(np.max(glcm))

    # Sum of Squares (Variance globala)
    mu_global = float(np.sum(i_grid * glcm))
    sum_of_squares = float(np.sum((i_grid - mu_global) ** 2 * glcm))

    # ── GRUP 2: distributii suma/diferenta (vectorizat) ───────────────────

    # p_{x+y}[k] = P(i,j) unde i+j = k, k in [0, 2*(n-1)]
    sum_indices  = (i_grid + j_grid).astype(np.int32)
    diff_indices = diff_grid.astype(np.int32)

    p_sum  = np.zeros(2 * (n - 1) + 1, dtype=np.float64)
    p_diff = np.zeros(n,               dtype=np.float64)
    np.add.at(p_sum,  sum_indices.ravel(),  glcm.ravel())
    np.add.at(p_diff, diff_indices.ravel(), glcm.ravel())

    # Sum Average: E[i+j]
    k_sum    = np.arange(len(p_sum), dtype=np.float64)
    sum_avg  = float(np.sum(k_sum * p_sum))

    # Sum Entropy: -suma(p_{x+y} * ln(p_{x+y}))
    nz_sum      = p_sum[p_sum > 0]
    sum_entropy = float(-np.sum(nz_sum * np.log(nz_sum)))

    # Difference Variance: Var(|i-j|)
    k_diff    = np.arange(n, dtype=np.float64)
    mu_diff   = float(np.sum(k_diff * p_diff))
    diff_var  = float(np.sum((k_diff - mu_diff) ** 2 * p_diff))

    # Difference Entropy: -suma(p_{x-y} * ln(p_{x-y}))
    nz_diff      = p_diff[p_diff > 0]
    diff_entropy = float(-np.sum(nz_diff * np.log(nz_diff)))

    return {
        "Autocorrelation"   : autocorr,
        "Dissimilarity"     : dissimilarity,
        "Homogeneity"       : homogeneity,
        "MaxProbability"    : max_prob,
        "SumOfSquares"      : sum_of_squares,
        "SumAverage"        : sum_avg,
        "SumEntropy"        : sum_entropy,
        "DifferenceVariance": diff_var,
        "DifferenceEntropy" : diff_entropy,
    }


def glcm_extended_features_for_pair(img, angles, distance, pair_name):
    """
    Calculeaza parametrii GLCM extinsi pentru o pereche de directii,
    identic metodologic cu glcm_features_for_pair din features.py:
    media GLCM pe cele 2 directii din pereche, apoi parametrii pe
    matricea medie.
    """
    glcm = graycomatrix(
        img,
        distances=[distance],
        angles=angles,
        levels=GLCM_LEVELS,
        symmetric=False,
        normed=True,
    )
    glcm_mean = (glcm[:, :, 0, 0] + glcm[:, :, 0, 1]) / 2
    params    = compute_glcm_extended_params(glcm_mean)

    suffix = f"_{pair_name}_d{distance}"
    return {f"{k}{suffix}": v for k, v in params.items()}


def glcm_extended_features(img):
    """
    Calculeaza cei 9 parametri GLCM extinsi pentru toate cele 4 perechi
    de directii si cele 3 distante.

    Total: 9 parametri * 4 directii * 3 distante = 108 parametri noi.

    Numele: Autocorrelation_0_180_d1, Dissimilarity_90_270_d10 etc.
    """
    feats = {}
    for pair_name, angles in GLCM_DIRECTION_PAIRS.items():
        for dist in GLCM_DISTANCES:
            feats.update(
                glcm_extended_features_for_pair(img, angles, dist, pair_name)
            )
    return feats


# ──────────────────────────── EXTRAGERE ─────────────────────────────────────

def extract_glcm_extended_features(filepath):
    """
    Incarca o imagine si calculeaza doar cei 108 parametri GLCM extinsi.
    Returneaza None daca imaginea nu poate fi procesata.

    Pentru setul complet de parametri (histograma + GLCM standard +
    GLRLM + GLCM extins), combinati cu main_glcm_extended.py.
    """
    img = load_grayscale(filepath)
    if img is None:
        return None
    return glcm_extended_features(img)