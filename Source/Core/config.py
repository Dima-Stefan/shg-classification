"""
config.py
Fisier central de configurare — singurul loc din proiect unde
sunt definite caile catre date, rezultate si modele.

Toate celelalte scripturi importa caile de aici in loc sa
le hardcodeze. Astfel proiectul functioneaza pe orice sistem
dupa clonare de pe GitHub, fara modificari suplimentare.

Structura asteptata a proiectului:
    project_root/
    ├── Data/                   # imaginile SHG (ignorat de git)
    ├── Results/
    │   ├── CSV/                # CSV-urile generate
    │   └── Plots/              # graficele generate
    └── Source/
        ├── core/               # pipeline de baza
        │   └── config.py       # acest fisier
        ├── classification/     # modele de clasificare
        └── analysis/           # selectie parametri + vizualizari
"""

from pathlib import Path

# ──────────────────────────── RADACINA PROIECTULUI ──────────────────────────

# Detectata automat relativ la locatia acestui fisier:
# Source/core/config.py -> Source/core -> Source -> ROOT
ROOT = Path(__file__).resolve().parent.parent.parent

# ──────────────────────────── DIRECTOARE PRINCIPALE ─────────────────────────

DATA_DIR    = ROOT / "Data"
RESULTS_DIR = ROOT / "Results"
SOURCE_DIR  = ROOT / "Source"

CSV_DIR   = RESULTS_DIR / "CSV"
PLOTS_DIR = RESULTS_DIR / "Plots"

# ──────────────────────────── CSV-URI ───────────────────────────────────────

# CSV-urile generate de pipeline-ul de extragere a parametrilor
CSV_FEATURES              = CSV_DIR / "features.csv"
CSV_FEATURES_GLRLM        = CSV_DIR / "features_glrlm.csv"
CSV_FEATURES_CAF          = CSV_DIR / "features_caf.csv"
CSV_FEATURES_GLCM_EXT     = CSV_DIR / "features_glcm_extended.csv"
CSV_FEATURES_PATCHES      = CSV_DIR / "features_patches_256_corners.csv"
CSV_FEATURES_NOTRIM       = CSV_DIR / "features_notrim.csv"

# ──────────────────────────── CONFIGURATIE SVM ──────────────────────────────

SVM_KERNEL       = "rbf"
SVM_RANDOM_STATE = 42
SVM_N_FOLDS      = 3
SVM_N_REPEATS    = 25

# ──────────────────────────── CONFIGURATIE GLCM ─────────────────────────────

GLCM_DISTANCES = [1, 10, 30]
GLCM_LEVELS    = 256

GLCM_DIRECTION_PAIRS = {
    "0_180"   : [0,              3.14159265],
    "90_270"  : [1.57079633,     4.71238898],
    "45_225"  : [0.78539816,     3.92699082],
    "135_315" : [2.35619449,     5.49778714],
}

# ──────────────────────────── CONFIGURATIE GLRLM ────────────────────────────

GLRLM_LEVELS = 256

# ──────────────────────────── CONFIGURATIE CAF ──────────────────────────────

CAF_THRESHOLD = 0  # pixeli cu intensitate > 0 sunt considerati semnal SHG

# ──────────────────────────── DIAGNOSTICE ───────────────────────────────────

DIAGNOSIS = {
    "72066_1" : "normal",
    "72066_2" : "malign",
    "72066_3" : "normal",

    "83064_1" : "normal",
    "83064_2" : "malign",
    "83064_3" : "malign",

    "89376_1" : "malign",
    "89376_2" : "malign",
    "89376_3" : "displazic",

    "102312_1": "normal",
    "102312_2": "malign",
    "102312_3": "malign",

    "149113_1": "displazic",
    "149113_2": "displazic",
    "149113_3": "displazic",

    "155820_1": "malign",
    "155820_2": "malign",
    "155820_3": "displazic",

    "156222_1": "normal",
    "156222_2": "normal",
    "156222_3": "normal",

    "182467_1": "normal",
    "182467_3": "normal",
}

# ──────────────────────────── VERIFICARE LA IMPORT ──────────────────────────

def check_structure():
    """
    Verifica ca structura de directoare asteptata exista.
    Apelata optional la rularea oricarui script principal.
    """
    missing = []
    for d in [DATA_DIR, CSV_DIR, PLOTS_DIR]:
        if not d.exists():
            missing.append(str(d))

    if missing:
        print(f"[config] Atentie: urmatoarele directoare nu exista:")
        for m in missing:
            print(f"  - {m}")
        print(f"[config] Radacina detectata: {ROOT}")
        return False
    return True


if __name__ == "__main__":
    print(f"Radacina proiectului : {ROOT}")
    print(f"Data                 : {DATA_DIR}")
    print(f"CSV                  : {CSV_DIR}")
    print(f"Plots                : {PLOTS_DIR}")
    print(f"Source               : {SOURCE_DIR}")
    check_structure()