# Geometry as Cause: The Δ-Metric Framework

**Paper:** *Geometry as Cause: The Δ-Metric Framework Explaining Why Linear Probe Difficulty Predicts Neural Scaling Exponents*

**Authors:** Md M. Zaman, A. Zaman, Neha Vinayak, Eht E. Sham — BITS Pilani, India

**Submitted to:** Transactions on Machine Learning Research (TMLR)

---

## Repository Structure

```
final_submission/
├── data/                           # Pre-computed experimental results
│   ├── results.csv                 # Main experiment: α + ΔM for 94 OpenML datasets
│   ├── intervention_results.csv    # Causal: Δα per dataset per intervention (551 trials)
│   ├── causal_tests.csv            # Statistical test results (Wilcoxon, sign acc, CI)
│   └── feature_deltas.csv          # Feature-level deltas across interventions
│
├── plots/                          # All paper figures (PNG, ready for LaTeX)
│   ├── fig1_dim_vs_alpha.png
│   ├── fig2_delta_vs_alpha.png
│   ├── fig3_regime_distributions.png
│   ├── fig4_feature_importance.png
│   ├── fig5_causal_summary.png
│   ├── fig6_distribution.png
│   ├── fig7_dose_response.png
│   ├── figA1_regularization_paradox.png
│   ├── figA2_causal_panel.png
│   └── figA3_before_after.png
│
├── src/
│   ├── nca_kaggle_experiment.py    # Main experiment (run on Kaggle, ~45 min CPU)
│   ├── nca_causal_intervention.py  # Causal intervention experiment (~2–3 hr CPU)
│   ├── generate_plots.py           # Generates Fig 1–4 from results.csv
│   └── generate_causal_plots.py    # Generates Fig 5–7 + appendix from CSVs
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Reproduce figures from pre-computed data

Figures 1–4 (main experiment):
```bash
python src/generate_plots.py --data data/results.csv
# Output → main_figures/ (PNG + PDF)
```

Figures 5–7 + Appendix (causal interventions):
```bash
# Copy CSVs to /content/ if on Colab, then:
python src/generate_causal_plots.py
# Output → causal_figures/ (PNG + PDF)
```

### 3. Re-run experiments from scratch (Kaggle)

```python
# Upload src/nca_kaggle_experiment.py to a Kaggle notebook
# Runtime: ~45 min CPU, no GPU needed
# Output: /kaggle/working/nca_out/results.csv
```

```python
# Upload src/nca_causal_intervention.py to a Kaggle notebook
# Runtime: ~2–3 hours CPU
# Output: intervention_results.csv, causal_tests.csv, feature_deltas.csv
```

---

## Key Results

| Metric | Value |
|--------|-------|
| Datasets accepted | 94 / 102 |
| Meta-model Spearman ρ (5-fold CV RF) | 0.334 (p = 0.0015) |
| Top predictor (RF importance) | Label noise (28.7%) |
| SC regime median α | 0.382 |
| Entangled regime median α | 0.168 |
| Scaling advantage SC/Entangled | 2.3× (KW p = 0.028) |
| Causal trials total | 551 across 95 datasets |
| Label-noise sign accuracy | 89–94% |
| Label-noise Wilcoxon p | < 10⁻¹⁴ |
| Dose-response Spearman ρ | −0.69 |

---

## Data

All datasets sourced from [OpenML](https://www.openml.org). The `data/` folder contains pre-computed results — you do **not** need to re-run the experiments to reproduce figures.

Raw datasets are not stored here; they are downloaded automatically by `nca_kaggle_experiment.py` via the `openml` Python package.

---
