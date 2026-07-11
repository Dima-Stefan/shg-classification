# SHG Image Classification

Clasificarea automata a tesutului biologic din imagini SHG (Second Harmonic Generation) in categorii patologice: **normal**, **malign** si **displazic**.

Universitatea Politehnica Bucuresti — 2025

---

## Structura proiectului

```
project_root/
├── Data/                          # Imaginile SHG (nu sunt incluse in repo)
├── Results/
│   ├── CSV/                       # CSV-urile generate de pipeline
│   └── Plots/                     # Graficele generate
└── Source/
    ├── core/                      # Pipeline-ul de baza
    │   ├── config.py              # Configurare centrala (cai, parametri)
    │   ├── utils.py               # Parsare fisiere, colectare imagini
    │   ├── features.py            # Parametri histograma + GLCM (64)
    │   ├── features_glrlm.py      # Parametri GLRLM (20 noi)
    │   ├── features_glcm_extended.py  # Parametri GLCM extinsi (108 noi)
    │   ├── features_caf.py        # CAF - Collagen Area Fraction
    │   ├── features_patches.py    # Extragere parametri din patch-uri 256x256
    │   ├── main.py                # Genereaza features.csv (64 parametri)
    │   ├── main_glrlm.py          # Genereaza features_glrlm.csv (84 parametri)
    │   ├── main_caf.py            # Genereaza features_caf.csv (85 parametri)
    │   └── main_glcm_extended.py  # Genereaza features_glcm_extended.csv (193 param)
    ├── classification/
    │   └── svm/
    │       ├── classify_svm_kfold.py        # Clasificare SVM principala
    │       └── classify_svm_kfold_color.py  # Scenarii HE vs non-HE
    └── analysis/
        ├── select_features.py          # Selectie parametri prin Lasso
        ├── select_features_rfe.py      # Selectie parametri prin RFE
        └── plot_correlation_heatmap.py # Heatmap corelatii Pearson
```

---

## Instalare

```bash
git clone https://github.com/<user>/shg-classification.git
cd shg-classification
python -m venv venv
venv\Scripts\activate        # Windows
# sau: source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

Adauga imaginile SHG in folderul `Data/` respectand conventia de denumire:
```
<cod_proba>_<sectiune>-<polarizare>_<canal>.tif
Exemplu: 102312_1-p0_ch00.tif
         102312HE_11-p20_ch02.tif
```

---

## Utilizare rapida

### 1. Extragere parametri

```bash
cd Source/core

# Parametri standard (histograma + GLCM, 64 parametri)
python main.py

# + GLRLM (84 parametri)
python main_glrlm.py

# + CAF (85 parametri)
python main_caf.py

# + GLCM extins (193 parametri)
python main_glcm_extended.py
```

### 2. Clasificare SVM

```bash
cd Source/classification/svm
python classify_svm_kfold.py
```

### 3. Selectie parametri

```bash
cd Source/analysis

# Lasso — pe orice CSV
python select_features.py --csv "../../Results/CSV/features_glrlm.csv"

# Lasso — doar parametrii GLRLM
python select_features.py --csv "../../Results/CSV/features_glrlm.csv" --group glrlm

# RFE — cu 25 parametri tinta
python select_features_rfe.py --csv "../../Results/CSV/features_glrlm.csv" --n-features 25
```

### 4. Heatmap corelatii

```bash
cd Source/analysis
python plot_correlation_heatmap.py
```

---

## Parametri extrasi

| Set | Fisier | Nr. parametri |
|-----|--------|---------------|
| Standard | `features.csv` | 64 |
| + GLRLM | `features_glrlm.csv` | 84 |
| + CAF | `features_caf.csv` | 85 |
| + GLCM extins | `features_glcm_extended.csv` | 193 |

### Tipuri de parametri

- **Histograma** (4): Mean, StDev, Skew, Kurt
- **GLCM standard** (60): ASM, Contrast, Correlation, IDM, Entropy × 4 directii × 3 distante
- **GLRLM** (20): SRE, LRE, GLN, RLN, RP × 4 directii
- **CAF** (1): Collagen Area Fraction
- **GLCM extins** (108): Autocorrelation, Dissimilarity, Homogeneity, MaxProbability, SumOfSquares, SumAverage, SumEntropy, DifferenceVariance, DifferenceEntropy × 4 directii × 3 distante

---

## Rezultate principale

| Model | Parametri | CV | Accuracy |
|-------|-----------|-----|----------|
| Decision Tree | 14 | Leave-2-Out | 55.1% |
| MLP | 14 | Leave-2-Out | 66.9% |
| SVM rbf | 14 | Leave-2-Out | 71.3% |
| SVM rbf | 64 (GLCM) | 25×3-Fold | 69.4% |
| SVM rbf | 84 (+ GLRLM) | 25×3-Fold | **71.8%** |

---

## Dependente

```
scikit-image, scikit-learn, scipy, numpy, pandas, matplotlib, tqdm
```

Vezi `requirements.txt` pentru versiunile exacte.

---

## Date

Imaginile SHG provin de la 8 pacienti (probe), fiecare cu sectiuni colorate (HE) si necolorate (non-HE), achizitionate la Universitatea Politehnica Bucuresti.

Din motive de confidentialitate, datele nu sunt incluse in repository.