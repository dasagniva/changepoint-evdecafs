# changepoint-evdecafs

EV-DeCAFS: Extreme Value Penalised Changepoint Detection and Classification via Fourier Probabilistic Neural Network.

## Overview

This repository implements a two-phase statistical pipeline for detecting and classifying changepoints in time series with heavy-tailed noise:

- **Phase I (Detection):** EV-DeCAFS — an AR(1)-aware penalised cost changepoint detector with adaptive EVT-based penalties derived from Generalised Pareto Distribution fits.
- **Phase II (Classification):** A Fourier Probabilistic Neural Network (FPNN) that classifies each detected changepoint as *sustained* or *recoiled*, trained on SMOTE-balanced feature vectors.

## Pipeline

```
Raw data
  └─> AR(1) estimation
  └─> EVT penalty (GPD fit → xi_t → alpha_t)
  └─> EV-DeCAFS recursion  (Phase I)
  └─> Feature extraction (delta_mu, persistence, phi_local, variance ratio)
  └─> Labelling + SMOTE balancing
  └─> FPNN training (Phase II)
  └─> Evaluation: MRL index, Hausdorff distance, cost-ratio sensitivity
```

## Datasets

- **Bitcoin:** Daily BTC-USD log prices (Yahoo Finance), 2014–2024.
- **Well-log:** Nuclear well-log response signal (Ruanaidh & Fitzgerald, 2012), 4050 observations. If the original dataset is unavailable, a synthetic surrogate is generated automatically.

## Project Structure

```
changepoint-evdecafs/
├── config/params.yaml          # All hyperparameters
├── src/
│   ├── data/loader.py          # Data loading & splitting
│   ├── phase1/                 # EV-DeCAFS core
│   ├── phase2/                 # FPNN + baselines
│   ├── evaluation/             # MRL, Hausdorff, sensitivity
│   ├── visualization/          # Publication figures
│   └── utils/logging_config.py
├── scripts/
│   ├── run_pipeline.py         # Full end-to-end run
│   └── run_phase1_comparison.py
├── results/figures/            # 300 DPI PDF/PNG outputs
├── results/tables/             # CSV result tables
├── notebooks/tester.ipynb      # Component testing notebook
└── tests/                      # Smoke tests
```

## Usage

```bash
# Install
pip install -e ".[dev]"

# Run full pipeline
python scripts/run_pipeline.py --dataset both

# Run only on Bitcoin
python scripts/run_pipeline.py --dataset bitcoin

# Skip baselines (faster iteration)
python scripts/run_pipeline.py --skip-baselines

# Custom config
python scripts/run_pipeline.py --config config/params.yaml --output-dir results/
```

## References

- Romano, G., Eckley, I. A., Fearnhead, P., & Rigaill, G. (2022). Fast online changepoint detection via functional pruning CUSUM statistics. *JMLR*.
- Ruanaidh, J. J. K. O., & Fitzgerald, W. J. (2012). *Numerical Bayesian Methods Applied to Signal Processing*. Springer.
