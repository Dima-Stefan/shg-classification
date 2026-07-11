"""
utils.py
Functii ajutatoare pentru parsarea numelor de fisiere SHG.

Conventie nume fisier:
    <cod_proba>_<sectiune>_<polarizare>_<canal>.tif
    Exemplu: 102312_1_p0_ch00.tif
             102312HE_11_p20_ch02.tif
"""

import re
from pathlib import Path


# Fisierele extra de achizitie care trebuie ignorate (nu sunt imagini)
IGNORED_SUFFIXES = {".txt", ".xml", ".json", ".ini", ".cfg", ".log"}
IGNORED_NAMES   = {"Thumbs.db", ".DS_Store"}

# Canalele de interes
VALID_CHANNELS = {"ch00", "ch02"}

# Fisierele de achizitie au de obicei un pattern diferit - le ignoram
# daca nu au exact formatul asteptat
FILENAME_PATTERN = re.compile(
    r"^(?P<probe>.+?)_"          # cod proba
    r"(?P<section>\d+)-"         # sectiune + cratima
    r"(?P<polar>p\d+|p_circ)_"   # polarizare
    r"(?P<channel>ch\d+)"        # canal
    r"\.tif$",
    re.IGNORECASE
)


def parse_filename(filepath: str | Path) -> dict | None:
    """
    Parseaza numele unui fisier .tif si returneaza un dictionar cu campurile extrase.
    Returneaza None daca fisierul nu respecta conventia sau trebuie ignorat.

    Campuri returnate:
        probe    : codul probei (ex: '102312' sau '102312HE')
        section  : sectiunea (ex: '1', '11')
        polar    : polarizarea (ex: 'p0', 'p_circ')
        channel  : canalul (ex: 'ch00')
        is_HE    : True daca proba e colorata cu HE
        filepath : calea completa catre fisier
    """
    path = Path(filepath)

    # Ignora fisierele non-imagine
    if path.suffix.lower() in IGNORED_SUFFIXES:
        return None
    if path.name in IGNORED_NAMES:
        return None

    match = FILENAME_PATTERN.match(path.name)
    if not match:
        return None

    channel = match.group("channel").lower()
    if channel not in VALID_CHANNELS:
        return None  # ignoram ch01 (fluorescenta)

    probe = match.group("probe")

    return {
        "probe"   : probe,
        "section" : match.group("section"),
        "polar"   : match.group("polar").lower(),
        "channel" : channel,
        "is_HE"   : probe.upper().endswith("HE"),
        "filepath": str(path.resolve()),
    }


def collect_images(data_root: str | Path) -> list[dict]:
    """
    Parcurge recursiv folderul data_root si colecteaza toate fisierele
    .tif valide (ch00 si ch02), ignorand restul.

    Returneaza o lista de dictionare (unul per fisier valid).
    """
    data_root = Path(data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Folderul nu exista: {data_root}")

    records = []
    for tif_file in sorted(data_root.rglob("*.tif")):
        info = parse_filename(tif_file)
        if info is not None:
            records.append(info)

    print(f"[utils] Fisiere valide gasite: {len(records)}")
    return records
