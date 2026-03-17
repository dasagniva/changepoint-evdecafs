# EV-DeCAFS

**Extreme Value DeCAFS** — a two-phase statistical pipeline for changepoint detection and classification in univariate time series with AR(1) noise and heavy-tailed excursions.

**Version:** v4.2 | **Python:** 3.10+

---

## Overview

EV-DeCAFS combines an EVT-adaptive penalised changepoint detector (Phase I) with a Fourier Probabilistic Neural Network classifier (Phase II):

- **Phase I — Detection:** EV-DeCAFS runs a dynamic programming recursion on an AR(1) cost function. The penalty is time-varying: local GPD fits to residual windows produce an EVI field ξ_t that inflates the penalty near heavy-tailed regions, suppressing spurious detections around outlier clusters.
- **Phase II — Classification:** Each detected changepoint is classified as *sustained* (true structural shift) or *recoiled* (transient excursion) by an FPNN trained on five handcrafted features (level shift magnitude, persistence, local AR coefficient, variance ratio, local EVI). Training labels are assigned by a **BOCPD oracle** (v4.2+).

```
Raw time series
  └─ AR(1) estimation (Yule-Walker)  →  phi, sigma_v², sigma_eta²
  └─ EVI field (GPD fits, w=50)      →  xi_t per time point
  └─ Adaptive penalty  alpha_t = alpha_0 * (1 + lambda_ev * |xi_t|)
       alpha_0 set by BIC sweep  C * log(n_train),  C auto-selected
  └─ DP recursion (500-point mu grid)  →  changepoint indices   [Phase I]
  └─ Feature extraction (5 features per CP)
  └─ BOCPD labelling oracle          →  sustained / recoiled labels
  └─ SMOTE balancing + FPNN training                             [Phase II]
  └─ Evaluation: MRL, Hausdorff, cost-ratio sensitivity, MC comparison
```

---

## Datasets

| Dataset | Type | n | Split | Notes |
|---------|------|---|-------|-------|
| `welllog` | Synthetic surrogate | 4 050 | 75/25 | Auto-generated if `data/raw/welllog.csv` absent; 12 true CPs |
| `oilwell` | Synthetic oilwell pressure | 4 000 | 75/25 | Step-change pressure regimes |
| `us_ip_growth` | FRED INDPRO monthly growth (2000–2026) | ~312 | 80/20 | Ground truth: NBER recession dates |

---

## Installation

```bash
git clone git@github.com:dasagniva/changepoint-evdecafs.git
cd changepoint-evdecafs
pip install -e ".[dev]"
```

---

## Usage

All scripts must be run from the `changepoint-evdecafs/` directory.

```bash
# Run full pipeline on well-log data
python scripts/run_pipeline.py --dataset welllog

# Run all three datasets
python scripts/run_pipeline.py --dataset all

# Skip baseline detectors (faster)
python scripts/run_pipeline.py --dataset all --skip-baselines

# Phase I penalty-variant comparison only
python scripts/run_phase1_comparison.py --dataset welllog

# Run tests
pytest tests/ -q
```

Output PDFs land in `results/figures/` and CSV tables in `results/tables/`.

---

## Repository Structure

```
changepoint-evdecafs/
├── config/params.yaml              # All hyperparameters (no hard-coded constants)
├── src/
│   ├── data/loader.py              # welllog, oilwell, FRED loaders
│   ├── phase1/
│   │   ├── ar1_model.py            # Yule-Walker AR(1) estimation
│   │   ├── evt_penalty.py          # GPD fits → EVI field → adaptive penalty
│   │   ├── decafs.py               # DP recursion (Algorithm 1)
│   │   ├── feature_extract.py      # 5-feature extractor (Algorithm 2)
│   │   ├── baseline_detectors.py   # PELT, BinSeg, BottomUp, Window, vanilla DeCAFS
│   │   └── hypersensitive_cpd.py   # BOCPD + CUSUM hypersensitive detector
│   ├── phase2/
│   │   ├── bocpd_labeller.py       # BOCPD labelling oracle (primary, v4.2+)
│   │   ├── labelling.py            # Traditional kappa_mu/kappa_S labeller (Algorithm 3)
│   │   ├── fpnn.py                 # FourierPNN classifier (Algorithms 4–5)
│   │   ├── smote_balance.py        # SMOTE class balancing
│   │   └── baselines.py            # LR, Isolation Forest, OC-SVM, MLP, GRU
│   ├── evaluation/
│   │   ├── mrl_index.py            # Mean Run Length + censored risk R̃
│   │   ├── hausdorff.py            # Hausdorff distance
│   │   ├── sensitivity.py          # Cost-ratio sensitivity analysis
│   │   ├── classification_metrics.py
│   │   ├── monte_carlo.py          # MC classifier comparison (B replications)
│   │   └── tail_diagnostics.py     # GPD tail summaries
│   ├── visualization/              # Publication-quality PDF figures
│   └── utils/logging_config.py
├── scripts/
│   ├── run_pipeline.py             # End-to-end orchestration
│   └── run_phase1_comparison.py    # Phase I variant comparison
├── notebooks/tester.ipynb          # 14-cell interactive component tester
└── tests/                          # 181 pytest tests across 6 files
```

---

## Key Results (v4.2, B=200 Monte Carlo)

| Classifier | Bal-Acc | MCC | AUC-ROC |
|------------|---------|-----|---------|
| **FPNN (proposed)** | **0.744 ± 0.241** | **0.235** | **0.792** |
| Logistic Regression | 0.616 ± 0.257 | 0.111 | 0.500 |
| GRU (RNN) | 0.525 ± 0.228 | 0.030 | 0.627 |
| Feedforward NN | 0.554 ± 0.243 | 0.000 | 0.224 |
| Isolation Forest | 0.349 ± 0.235 | −0.181 | 0.272 |
| One-Class SVM | 0.284 ± 0.182 | −0.292 | 0.060 |

Synthetic series: n=2 000, 8 sustained CPs, 15 outlier spikes, AR(1) φ=0.5, σ_v=2 000.

---

## Configuration

All tunable constants live in `config/params.yaml`. Key sections:

| Section | Key parameters |
|---------|---------------|
| `phase1` | `bic_sweep_values`, `evt_sensitivity_lambda_ev`, `window_halfwidth_w` |
| `bocpd` | `threshold=0.3`, `tolerance_fraction=0.02`, NIG prior hyperparameters |
| `labelling` | `kappa_mu_percentile=75`, `kappa_S=0.5` |
| `fpnn` | `J_harmonics=10`, `scaling_range=[-0.5, 0.5]` |
| `monte_carlo` | `B=200`, synthetic series parameters |
| `evaluation` | cost grids for MRL sensitivity analysis |

---

## References

- Romano, G., Eckley, I. A., Fearnhead, P., & Rigaill, G. (2022). Fast online changepoint detection via functional pruning CUSUM statistics. *JMLR*.
- Adams, R. P., & MacKay, D. J. C. (2007). Bayesian online changepoint detection. *arXiv:0710.3742*.
- Pickands, J. (1975). Statistical inference using extreme order statistics. *Annals of Statistics*.
