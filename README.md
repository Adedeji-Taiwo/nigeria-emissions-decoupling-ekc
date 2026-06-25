# Agricultural Emission Drivers, Decoupling Dynamics, and EKC Evidence in Nigeria  
### LMDI · Tapio Decoupling · ARDL‑EKC · Fully Reproducible

[![DOI](https://img.shields.io/badge/DOI-10.2139%2Fssrn.6640038-blue)](https://dx.doi.org/10.2139/ssrn.6640038)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)

This repository contains the complete computational pipeline for the paper:

> **Beyond Scale Effects: Agricultural Emission Drivers, Decoupling Dynamics, and Environmental Kuznets Curve Evidence in Nigeria**  
> *Taiwo Adedeji Michael, Qudus Oladeji Olamide, and Adeyeye Daniel Oludayo*  
> Preprint available at https://dx.doi.org/10.2139/ssrn.6640038

---

## What this repo does

The study decomposes Nigeria’s agricultural GHG emission trajectory (1990–2021) using three complementary approaches:

- **Logarithmic Mean Divisia Index (LMDI)** – isolates the contribution of five Kaya‑identity drivers.
- **Tapio decoupling elasticity analysis** – classifies emission‑output relationships.
- **Environmental Kuznets Curve (EKC) via ARDL bounds testing** – estimates long‑run income–emission relationships.

The LMDI and Tapio calculations are provided in Excel, while the EKC analysis is fully reproducible via Python scripts.

---

## Repository structure

```bash
.
├── data/
│   ├── nigeria_agric_dataset.xlsx
├── ekc/
│   ├── nigeria_agric_data.py
│   ├── ekc_preliminary.py
│   ├── ekc_unit_root_adf.py
│   ├── ekc_unit_root_pp.py
│   ├── ekc_unit_root_za.py
│   ├── ekc_ardl_lag_selection.py
│   ├── ekc_bounds_test.py
│   ├── ekc_error_correction.py
│   ├── ekc_longrun_coefficients.py
│   ├── ekc_shortrun_dynamics.py
│   ├── ekc_cubic_robustness.py
│   ├── ekc_diagnostics.py
│   ├── ekc_plot_longrun_curve.py
│   ├── results/
│   └── figures/
├── lmdi_tapio/
│   └── lmdi_tapio_analysis.xlsx
├── all_figures/
├── requirements.txt
└── README.md
```

### Note

- The figures/ folder contains all figures used in the manuscript (conceptual framework, LMDI decomposition plots, Tapio decoupling index chart, EKC curve, etc.). They are provided for direct reference alongside the code and data.

---

## Installation & Requirements

1. Clone the repository:

```bash
git clone https://github.com/Adedeji-Taiwo/nigeria-agric-emissions.git
cd nigeria-agric-emissions
```

2. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate          # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

3. Data availability:

All data required by the EKC scripts is contained directly inside `ekc/nigeria_agric_data.py`.

---

## How to reproduce the EKC results

Run the scripts in the `ekc/` folder in the following order:

| Step | Script                          | Description |
|------|--------------------------------|-------------|
| 1    | ekc_preliminary.py             | Descriptive statistics and correlations |
| 2    | ekc_unit_root_adf.py           | ADF unit root tests |
| 3    | ekc_unit_root_pp.py            | Phillips‑Perron tests |
| 4    | ekc_unit_root_za.py            | Zivot‑Andrews structural break tests |
| 5    | ekc_ardl_lag_selection.py      | ARDL lag selection (AIC) |
| 6    | ekc_bounds_test.py             | Bounds test for cointegration |
| 7    | ekc_error_correction.py        | Error‑correction model |
| 8    | ekc_longrun_coefficients.py    | Long‑run EKC coefficients |
| 9    | ekc_shortrun_dynamics.py       | Short‑run dynamics |
| 10   | ekc_cubic_robustness.py        | Quadratic vs cubic EKC |
| 11   | ekc_diagnostics.py             | Residual diagnostic tests |
| 12   | ekc_plot_longrun_curve.py      | EKC curve figure |

---

## Key outputs

### CSV outputs (ekc/results/)

- descriptive_stats.csv
- correlation_matrix.csv
- adf_unit_root.csv
- pp_unit_root.csv
- za_unit_root.csv
- bounds_test_result.csv
- ect_result.csv
- longrun_coefficients.csv
- ekc_shape_summary.csv
- shortrun_dynamics.csv
- cubic_robustness.csv
- diagnostics.csv

### Figures (ekc/figures/)

- ekc_ardl_curve.png — ARDL‑implied long‑run EKC curve

---

## Citation

```bibtex
@article{michael2025beyond,
  title   = {Beyond Scale Effects: Agricultural Emission Drivers, Decoupling Dynamics, and Environmental Kuznets Curve Evidence in Nigeria},
  author  = {Michael, Taiwo Adedeji and Olamide, Qudus Oladeji and Oludayo, Adeyeye Daniel},
  year    = {2025},
  doi     = {10.2139/ssrn.6640038},
  note    = {Preprint available at https://dx.doi.org/10.2139/ssrn.6640038}
}
```

---

## Contact

Michael.taiwoadedeji@um6p.ma

---

## License

MIT License
