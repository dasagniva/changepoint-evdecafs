# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (from changepoint-evdecafs/ directory)
pip install -e ".[dev]"

# Run full pipeline on well-log data
python scripts/run_pipeline.py --dataset welllog

# Run full pipeline on all datasets, skip baseline detectors
python scripts/run_pipeline.py --dataset all --skip-baselines

# Run Phase I penalty-variant comparison only
python scripts/run_phase1_comparison.py --dataset welllog

# Run all tests
pytest tests/ -q

# Run a single test file
pytest tests/test_phase1.py -v

# Run a single test by name
pytest tests/test_phase1.py::test_ar1_estimation -v

# Launch interactive notebook tester
jupyter notebook notebooks/tester.ipynb
```

All pipeline scripts must be run from the `changepoint-evdecafs/` directory (they insert the repo root into `sys.path`).

## Architecture

This is a **two-phase statistical pipeline** for changepoint detection and classification in AR(1) time series with heavy-tailed noise.

### Data Flow

```
Raw time series
  └─ Phase I: EV-DeCAFS detection
       ├─ AR(1) estimation (Yule-Walker) → phi, sigma_v^2, sigma_eta^2
       ├─ EVI field (GPD fits on local windows w=50) → xi_t per timepoint
       ├─ Adaptive penalty schedule: alpha_t = alpha_0 * (1 + lambda_ev * |xi_t|)
       │    alpha_0 set by BIC sweep: C * log(n_train), C tuned to ~expected CPs
       └─ DP recursion on 500-point mu grid → changepoint indices
  └─ Phase I→II Bridge
       ├─ Feature extraction: 5 features per CP (delta_mu, persistence S, phi_local, variance ratio V, xi_local)
       └─ Labelling: sustained (|delta_mu|>kappa_mu AND S>0.5) vs recoiled
  └─ Phase II: FPNN classification
       ├─ SMOTE balancing on training features
       ├─ FourierPNN: Fejér-weighted Fourier density estimator per class per feature
       └─ Baselines: Logistic Regression, Isolation Forest, OC-SVM, MLP
```

### Key Source Files

| File | Role |
|------|------|
| `src/phase1/ar1_model.py` | Yule-Walker AR(1) estimation; `compute_bic_penalty()` |
| `src/phase1/evt_penalty.py` | GPD fits → EVI field; `compute_adaptive_penalty()`, `compute_exceedance_count_penalty()` |
| `src/phase1/decafs.py` | DP recursion `ev_decafs()` — the core Algorithm 1 |
| `src/phase1/feature_extract.py` | 5-feature extractor per changepoint (delta_mu, S, phi_local, V, xi_local) |
| `src/phase1/baseline_detectors.py` | PELT, BinSeg, BottomUp, Window, vanilla DeCAFS via ruptures |
| `src/phase2/labelling.py` | Sustained/recoiled labelling (Algorithm 3) |
| `src/phase2/fpnn.py` | `FourierPNN` classifier (Algorithms 4–5) |
| `src/phase2/baselines.py` | `get_baselines()`, `train_and_evaluate_all()` |
| `src/data/loader.py` | Well-log, oilwell, US IP growth loaders; auto-generates synthetic surrogate if CSV absent; flexible FRED CSV column parsing (`DATE` or `observation_date`) |
| `config/params.yaml` | **All hyperparameters** — never hard-coded in source |
| `scripts/run_pipeline.py` | End-to-end orchestration (603 lines); `run_phase1()`, `run_phase2_train()`, `run_phase2_test()`, `run_mrl_analysis()`, `_make_figures()` |

### Configuration

All tunable constants live in `config/params.yaml`. Key sections:
- `phase1`: BIC sweep (`tune_bic`, `bic_sweep_values`), EVT sensitivity (`evt_sensitivity_lambda_ev`, `window_halfwidth_w`, `gpd_percentile_q0`)
- `labelling`: `kappa_mu_percentile`, `kappa_S`, `window_L`
- `fpnn`: `J_harmonics`, `scaling_range`
- `evaluation`: cost grids for MRL sensitivity analysis
- `monte_carlo`: `B` (replications), synthetic series parameters

### Datasets

Three datasets are supported (`--dataset [welllog|oilwell|us_ip_growth|all]`):

| Dataset | Type | Split | Notes |
|---------|------|-------|-------|
| `welllog` | Synthetic surrogate (4050 obs, 12 true CPs) | 75/25 | Auto-generated if `data/raw/welllog.csv` absent |
| `oilwell` | Synthetic oilwell pressure series (4000 obs) | 80/20 | Auto-generated |
| `us_ip_growth` | FRED INDPRO monthly growth rate (2000–2026) | 80/20 (train through 2023-12) | Ground truth: NBER recession dates; fallback to direct FRED CSV if no API key |

### DP Implementation Notes

- The level `mu` is discretised on a 500-point grid spanning `[min(y)−2σ, max(y)+2σ]`
- Backtrack pointers stored as `int16` (shape `n × 500`) — sufficient since 500 < 32767
- Typical runtime: ~1.6 s for n=3037 (training), ~0.56 s for n=1013 (test)
- EVI field uses method-of-moments GPD estimation (fast, ~0.15 s total); MLE was dropped due to 130× slowdown

### Output

Results land in `results/figures/` (300 DPI PDFs) and `results/tables/` (CSVs). Key outputs per dataset `{ds}`:

**Figures:**
- `{ds}_train_run_chart.pdf` / `{ds}_test_run_chart.pdf` — run charts with CP labels
- `{ds}_phase2_classification.pdf` — THE key figure (sustained=green, recoiled=orange)
- `{ds}_phase1_detector_comparison.pdf` — 6-panel detector comparison grid
- `{ds}_phase1_multimetric.pdf` — grouped bar chart (always generated, even with 0 test CPs)
- `{ds}_tail_diagnostics_train.pdf` — 3-panel tail diagnostics (test only if n_test > 200)
- `{ds}_fig_mc_classifier_comparison.pdf` — 2×2 violin plot (MC runs for ALL datasets)
- `us_ip_growth_annotated.pdf` — IP index with NBER recession bands (US IP only)

**Tables:**
- `{ds}_phase1_detector_comparison.csv` — MRL / Hausdorff / risk per detector
- `{ds}_monte_carlo_all_classifiers.csv` — BA / MCC / AUC-ROC / Brier / Kappa over 500 MC runs
- `{ds}_monte_carlo_coverage.csv` — full per-metric MC aggregation
- `{ds}_bic_sweep.csv` — C vs n_detected sweep used to select `alpha_0`
- `{ds}_tail_diagnostics_train.csv` / `{ds}_tail_summary_train.csv` — per-window and aggregate tail stats

### Testing

158 pytest tests across 6 files in `tests/`. Visualisation tests use the `Agg` backend (headless). `test_smoke.py` runs a tiny end-to-end pipeline in < 1 s.
