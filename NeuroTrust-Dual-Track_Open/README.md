# NeuroTrust–Dual-Track — Reproducibility Package (Open version)

Supplementary code, **aggregated results, and de-identified data** for:

> *Neuro-Trust and System Trust: A Dual-Track Trust Structure and Antecedent Patterns in EEG-BCI Attention Training*
> Manuscript ID **electronics-4596113**, *Electronics* (MDPI).

## What this repository provides
- Complete analysis scripts: CFA/SEM (IBM SPSS AMOS), Harman common-method-bias test, Elastic Net, XGBoost, SHAP.
- A **de-identified construct-score dataset for all 274 participants**.
- Aggregated statistical outputs (CSV/JSON) behind every reported number.
- The 36 measurement items and the item-to-construct map.

## Repository layout
```
data/
  deidentified_construct_scores.csv   274 rows; PC, FS, PFE, WCE, PSC, PARV, NT, ST, RI.
                                      No IDs and no item-level responses.
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
- **De-identified construct scores for all 274 participants are included** in `data/deidentified_construct_scores.csv`.
- The **raw item-level responses remain withheld** because they contain sensitive
  attention/mental-state information; the item-level data needed to rebuild the scores are
  available from the corresponding author upon reasonable request, subject to ethics and confidentiality.

## Reproduction order
1. `python -m pip install -r requirements.txt`
2. Harman CMB test .................. `scripts/analysis/harman.py`
3. CFA/SEM & bootstrap .............. `scripts/amos_sem/*.ps1` (requires IBM SPSS AMOS)
4. ML data preparation .............. `scripts/machine_learning/ml_1_prep.py`
5. Nested CV (Elastic Net + XGBoost) . `ml_3_nested.py`
6. Out-of-fold SHAP ................. `ml_9_oof_shap.py`
7. Figures .......................... `ml_10_oof_fig.py`, `scripts/analysis/draw_fig1.py`, `draw_fig3_dual.py`

## Methodological notes
- SHAP values are **aggregated across held-out outer test folds (out-of-fold)**, not computed from a single final refit.
- SEM and ML use the **same 274 participants**; the ML stage provides multi-method **convergent** evidence, not independent validation.

## License
CC BY 4.0 — see [LICENSE]. Please cite the manuscript if you use these materials.
