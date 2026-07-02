# geometry-as-cause: Geometric Learnability Signal for Neural Scaling Laws

**Predicting Neural Scaling Exponents from Linear Probe Geometry***

**Authors:** Md M. Zaman, A. Zaman, Neha Vinayak, Eht E. Sham — BITS Pilani, India

**Submitted to:** Transactions on Machine Learning Research (TMLR)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Abstract

This paper asks a concrete question: given a tabular classification dataset, can you predict how fast a neural network will learn from it as the training set grows? We show that the answer is largely determined by the geometric structure of the class boundaries. We introduce the **Δ-metric** (ΔM = FR × S̃, the product of the Fisher Ratio and a shifted Silhouette Score) as a low-cost geometric descriptor that predicts the power-law scaling exponent α across 94 OpenML datasets. Datasets partition naturally into three regimes — Separable-Coherent (SC), Separable-Incoherent (SI), and Entangled — with a 2.3× median α gap between SC and Entangled (Kruskal–Wallis p = 0.028). A Random Forest meta-model achieves Spearman ρ = 0.334 (p = 0.0015) predicting α from geometric features alone. Critically, six controlled interventions across 551 trials demonstrate that manipulating the geometry causally shifts α in the predicted direction, with label-noise interventions achieving 89–94% directional accuracy and Wilcoxon p < 10⁻¹⁴.

---

## Key Findings

| Metric | Value |
|--------|-------|
| Datasets accepted | 94 / 102 |
| Meta-model Spearman ρ (5-fold CV RF) | 0.334 (p = 0.0015) |
| Top predictive feature (RF importance) | Label noise (28.7%) |
| SC regime median α | 0.382 |
| SI regime median α | 0.327 |
| Entangled regime median α | 0.168 |
| SC / Entangled α ratio | 2.3× (KW p = 0.028) |
| Intervention trials | 551 across 95 datasets |
| Label-noise sign accuracy | 89–94% |
| Label-noise Wilcoxon p | < 10⁻¹⁴ |
| Dose-response Spearman ρ | −0.69 |

---

## Repository Structure

```
geometry-as-cause/
├── data/                              # Pre-computed experimental results
│   ├── results.csv                    # Main experiment: α + ΔM for 94 OpenML datasets
│   ├── intervention_results.csv       # Δα per dataset per intervention (551 trials)
│   ├── causal_tests.csv               # Statistical test results (Wilcoxon, sign acc, CI)
│   └── feature_deltas.csv             # Feature-level deltas across interventions
│
├── plots/                             # All paper figures (PNG + PDF)
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
│   ├── __init__.py
│   ├── scaling_experiment.py          # Main experiment: α estimation + geometric features
│   ├── causal_intervention.py         # Causal intervention experiment
│   ├── generate_plots.py              # Figures 1–4 from results.csv
│   └── generate_causal_plots.py       # Figures 5–7 + appendix from intervention CSVs
│
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md
```

---

## Quickstart

### Install dependencies

```bash
pip install -r requirements.txt
```

### Reproduce all figures from pre-computed data

All pre-computed result CSVs are in `data/` — you do not need to re-run the experiments to generate figures.

**Figures 1–4** (main experiment — geometry vs α):

```bash
python src/generate_plots.py --data data/results.csv
# Outputs → main_figures/
```

**Figures 5–7 and Appendix** (causal interventions):

```bash
# Copy the CSVs to the expected location first, or edit CONTENT in the script
cp data/intervention_results.csv /content/
cp data/causal_tests.csv /content/
cp data/feature_deltas.csv /content/
python src/generate_causal_plots.py
# Outputs → causal_figures/
```

---

## Re-running Experiments from Scratch

Datasets are downloaded automatically via the `openml` Python package; they are not stored in this repository. All 94–102 datasets are publicly available on [OpenML](https://www.openml.org).

**Main scaling experiment** — computes α and ΔM for all datasets:

```bash
python src/scaling_experiment.py
# Runtime: ~45 min on a CPU-only machine
# Output: ./nca_out/results.csv  (plus results_checkpoint.csv every 10 datasets)
```

**Causal intervention experiment** — six interventions × ~95 datasets:

```bash
python src/causal_intervention.py
# Runtime: ~2–3 hours on CPU
# Output: ./causal_out/intervention_results.csv
#         ./causal_out/causal_tests.csv
#         ./causal_out/feature_deltas.csv
#         ./causal_out/summary.json
```

Both scripts use `MASTER_SEED = 42` throughout. The scaling experiment averages MLP learning curves over 3 seeds; the intervention experiment uses 2 seeds per size point to keep wall-clock time manageable.

---

## Environment / Requirements

Tested on Python 3.10 (Kaggle notebook environment) and Python 3.11 (Google Colab). All dependencies are standard scientific Python packages:

```
numpy>=1.24
pandas>=1.5
scipy>=1.10
scikit-learn>=1.2
matplotlib>=3.7
seaborn>=0.12
openml>=0.14
imbalanced-learn>=0.10
tqdm>=4.64
```

Install with:

```bash
pip install -r requirements.txt
```

---

## Citation

If you use this code or the pre-computed results, please cite:

```bibtex
@article{zaman2025geometry,
  title   = {Geometry as Learnability Signal: The $\Delta$-Metric Framework},
  author  = {Zaman, Md M. and Zaman, A. and Vinayak, Neha and Sham, Eht E.},
  journal = {Transactions on Machine Learning Research},
  year    = {2025},
  url     = {https://github.com/Zaman208/geometry-as-cause}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
