"""
features_glrlm.py
Extensie a features.py care adauga 20 de parametri noi din Gray-Level
Run-Length Matrix (GLRLM), propusi de profesor.

GLRLM masoara lungimea "rulajelor" (secvente consecutive de pixeli cu
aceeasi intensitate) pe o directie data. Spre diferenta de GLCM, GLRLM
NU are conceptul de "distanta" -- o singura matrice per directie
captureaza deja toate lungimile de rulaj posibile, de la 1 pana la
marginea imaginii.

Parametrii calculati pentru fiecare din cele 4 perechi de directii
(0_180, 90_270, 45_225, 135_315) -- consistent cu GLCM, prin medierea
GLRLM-urilor celor 2 directii din pereche:
    SRE (Short Run Emphasis)        -- accentueaza rulajele scurte
    LRE (Long Run Emphasis)         -- accentueaza rulajele lungi
    GLN (Gray-Level Non-Uniformity) -- non-uniformitatea nivelurilor de gri
    RLN (Run-Length Non-Uniformity) -- non-uniformitatea lungimilor de rulaj
    RP  (Run Percentage)            -- procentul de pixeli in rulaje

Total: 5 parametri x 4 perechi de directii = 20 parametri noi.

Numele parametrilor: ex. SRE_0_180, LRE_90_270, GLN_45_225 etc.
(fara sufix de distanta, pentru ca GLRLM nu are acest concept)

Acest fisier este INDEPENDENT de features.py si features_corr_std.py --
nu importa nimic din ele, poate fi testat separat.
"""

import numpy as np
from skimage import io


# ──────────────────────────── CONFIGURARE ───────────────────────────────────

# Perechile de directii opuse, consistent cu GLCM (in grade, nu radiani,
# pentru claritate -- 0/90/45/135 plus opusele lor 180/270/225/315
# sunt echivalente geometric cu directiile de baza pentru run-length)
GLRLM_DIRECTION_PAIRS = {
    "0_180"   : 0,
    "90_270"  : 90,
    "45_225"  : 45,
    "135_315" : 135,
}

# Numarul de nivele de gri -- identic cu GLCM_LEVELS din features.py,
# pentru consistenta intre cei doi seturi de parametri
GLRLM_LEVELS = 256


# ──────────────────────────── INCARCARE IMAGINE ─────────────────────────────

def load_grayscale(filepath):
    """
    Incarca o imagine .tif si o aduce la grayscale uint8 (0-255).
    Identic cu features.py. Returneaza None daca fisierul nu poate
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
        print(f"[features_glrlm] EROARE la citirea {filepath}: {e}")
        return None


# ──────────────────────────── CONSTRUCTIE GLRLM ─────────────────────────────

def _process_line(line, glrlm):
    """
    Parcurge o linie 1D de pixeli si actualizeaza glrlm cu rulajele
    gasite. Un rulaj = o secventa consecutiva de pixeli cu aceeasi
    valoare de intensitate.
    """
    n = len(line)
    i = 0
    while i < n:
        val = line[i]
        run_length = 1
        while i + run_length < n and line[i + run_length] == val:
            run_length += 1
        glrlm[val, run_length - 1] += 1
        i += run_length


def compute_glrlm(img, angle_deg, levels=GLRLM_LEVELS):
    """
    Calculeaza Gray-Level Run-Length Matrix pentru o imagine 2D,
    pe o singura directie (0, 45, 90 sau 135 grade).

    glrlm[i, j] = numarul de rulaje de lungime (j+1) avand nivelul
    de gri i, pe directia specificata.

    Pentru 0/90 (orizontal/vertical) parcurgem randuri/coloane direct.
    Pentru 45/135 (diagonale) folosim np.diagonal pe toate offset-urile
    posibile, ca sa acoperim toata imaginea.
    """
    img = img.astype(np.int64)
    h, w = img.shape
    max_run_length = max(h, w)

    glrlm = np.zeros((levels, max_run_length), dtype=np.float64)

    if angle_deg == 0:
        for row in range(h):
            _process_line(img[row, :], glrlm)
    elif angle_deg == 90:
        for col in range(w):
            _process_line(img[:, col], glrlm)
    elif angle_deg == 45:
        # Diagonala principala: stanga-jos -> dreapta-sus
        for offset in range(-(h - 1), w):
            diag = np.diagonal(img, offset=offset)
            if len(diag) > 0:
                _process_line(diag, glrlm)
    elif angle_deg == 135:
        # Diagonala secundara: oglindim imaginea pe orizontala, apoi
        # extragem diagonalele -- echivalent cu directia 135°
        img_flipped = np.fliplr(img)
        for offset in range(-(h - 1), w):
            diag = np.diagonal(img_flipped, offset=offset)
            if len(diag) > 0:
                _process_line(diag, glrlm)
    else:
        raise ValueError(f"[features_glrlm] Unghi nesuportat: {angle_deg}")

    return glrlm


# ──────────────────────────── PARAMETRI GLRLM ───────────────────────────────

def compute_glrlm_params(glrlm):
    """
    Calculeaza cei 5 parametri standard dintr-o matrice GLRLM:
        SRE, LRE, GLN, RLN, RP

    Formulele sunt cele standard din literatura (Galloway 1975 si
    extensiile uzuale folosite si in pyradiomics):
        n_runs = numarul total de rulaje gasite
        j      = lungimea rulajului (1, 2, 3, ...)
    """
    n_runs = glrlm.sum()
    if n_runs == 0:
        return {"SRE": 0.0, "LRE": 0.0, "GLN": 0.0, "RLN": 0.0, "RP": 0.0}

    levels, max_run = glrlm.shape
    j_vals = np.arange(1, max_run + 1, dtype=np.float64)

    # SRE: Short Run Emphasis -- pondereaza rulajele scurte mai mult
    sre = float(np.sum(glrlm / (j_vals[np.newaxis, :] ** 2)) / n_runs)

    # LRE: Long Run Emphasis -- pondereaza rulajele lungi mai mult
    lre = float(np.sum(glrlm * (j_vals[np.newaxis, :] ** 2)) / n_runs)

    # GLN: Gray-Level Non-Uniformity -- cat de neuniforma e distributia
    # rulajelor pe nivelurile de gri (valoare mica = distributie uniforma)
    gln = float(np.sum(glrlm.sum(axis=1) ** 2) / n_runs)

    # RLN: Run-Length Non-Uniformity -- cat de neuniforma e distributia
    # rulajelor pe lungimi (valoare mica = lungimi variate uniform)
    rln = float(np.sum(glrlm.sum(axis=0) ** 2) / n_runs)

    # RP: Run Percentage -- numarul de rulaje relativ la numarul total
    # de pixeli implicati in acele rulaje (valoare mare = rulaje scurte
    # si numeroase; valoare mica = rulaje lungi si putine)
    total_pixels = float(np.sum(glrlm * j_vals[np.newaxis, :]))
    rp = float(n_runs / total_pixels) if total_pixels > 0 else 0.0

    return {"SRE": sre, "LRE": lre, "GLN": gln, "RLN": rln, "RP": rp}


def glrlm_features_for_pair(img, angle_deg, pair_name):
    """
    Calculeaza parametrii GLRLM pentru o pereche de directii opuse,
    facand media celor doua matrici GLRLM individuale -- consistent
    cu modul in care features.py face media GLCM pe perechi.

    Pentru run-length, directiile opuse (ex. 0° si 180°) produc
    matrici GLRLM identice (parcurgerea unei linii de la dreapta la
    stanga gaseste aceleasi rulaje ca de la stanga la dreapta), deci
    practic calculam o singura matrice per pereche, dar pastram
    structura de "pereche" pentru consistenta nomenclatorului cu GLCM.
    """
    glrlm = compute_glrlm(img, angle_deg)
    params = compute_glrlm_params(glrlm)

    suffix = f"_{pair_name}"
    return {f"{k}{suffix}": v for k, v in params.items()}


def glrlm_features(img):
    """
    Calculeaza parametrii GLRLM pentru toate cele 4 perechi de directii.

    Numele parametrilor: SRE_0_180, LRE_90_270, GLN_45_225,
    RLN_135_315, RP_0_180 etc. (20 parametri in total, fara sufix
    de distanta).
    """
    feats = {}
    for pair_name, angle_deg in GLRLM_DIRECTION_PAIRS.items():
        feats.update(glrlm_features_for_pair(img, angle_deg, pair_name))
    return feats


# ──────────────────────────── EXTRAGERE COMPLETA ────────────────────────────

def extract_glrlm_features(filepath):
    """
    Incarca o imagine si calculeaza doar cei 20 de parametri GLRLM.
    Returneaza None daca imaginea nu poate fi procesata.

    Acest fisier NU calculeaza parametrii histograma/GLCM existenti --
    pentru un set complet de parametri (histograma + GLCM + GLRLM),
    combinati rezultatul cu cel din features.py sau features_corr_std.py.
    """
    img = load_grayscale(filepath)
    if img is None:
        return None

    return glrlm_features(img)