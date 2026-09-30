# NeuroTrust–Dual-Track — Reproducibility Package (Conservative version)

Supplementary code and **aggregated results** for:

> *Neuro-Trust and System Trust: A Dual-Track Trust Structure and Antecedent Patterns in EEG-BCI Attention Training*
> Manuscript ID **electronics-4596113**, *Electronics* (MDPI).

## What this repository provides
- Complete analysis scripts: CFA/SEM (IBM SPSS AMOS), Harman common-method-bias test, Elastic Net, XGBoost, SHAP.
- Aggregated statistical outputs (CSV/JSON) behind every reported number.
- The 36 measurement items and the item-to-construct map.
- Step-by-step reproduction notes.

## Repository layout
```
scripts/
  amos_sem/          PowerShell drivers automating AMOS: chi-square difference, model A/B,
                     partial-mediation test, bootstrap.
  analysis/          Harman CMB test; SEM/SHAP figure scripts.
  machine_learning/  data preparation, repeated nested CV (Elastic Net + XGBoost),
                     out-of-fold SHAP, figures.
results/
  CFA/  SEM/  MachineLearning/   aggregated outputs (CSV); harman_cmb.json.
measurement/     36 items and construct map.
```

## Data availability (please read)
The **raw survey responses and item-level data are NOT included**, because they contain sensitive
information about participants' attention/mental-state experiences. This package therefore
publishes the **code and aggregated results only**.

To reproduce the analyses:
1. `python -m pip install -r requirements.txt`
2. Request the **de-identified dataset** from the corresponding author (reasonable request,
   subject to ethics and confidentiality).
3. Run the scripts in the order below.

## Reproduction order
1. Harman CMB test .................. `scripts/analysis/harman.py`
2. CFA/SEM & bootstrap .............. `scripts/amos_sem/*.ps1` (requires IBM SPSS AMOS)
3. ML data preparation .............. `scripts/machine_learning/ml_1_prep.py`
4. Nested CV (Elastic Net + XGBoost) . `ml_3_nested.py`
5. Out-of-fold SHAP ................. `ml_9_oof_shap.py`
6. Figures .......................... `ml_10_oof_fig.py`, `scripts/analysis/draw_fig1.py`, `draw_fig3_dual.py`

The CSVs in `results/` correspond to each step.

## Methodological notes
- SHAP values are **aggregated across held-out outer test folds (out-of-fold)**, not computed from a single final refit.
- SEM and ML use the **same 274 participants**; the ML stage provides multi-method **convergent** evidence, not independent validation.

## License
CC BY 4.0 — see [LICENSE](LICENSE). Please cite the manuscript if you use these materials.
