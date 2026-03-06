# EV-DeCAFS: Project Summary

## Overview

**EV-DeCAFS** (Extreme Value DeCAFS) is a two-phase statistical pipeline for
changepoint detection and classification in univariate time series with AR(1)
noise. The central innovation is an adaptive penalty schedule derived from
Extreme Value Theory (EVT): local Generalised Pareto Distribution (GPD) fits
produce a time-varying shape parameter (EVI field ξ_t) that inflates the
changepoint penalty in regions of heavy-tailed behaviour, preventing spurious
detections near outlier clusters. Detected changepoints are then classified as
*sustained* (true structural shifts) or *recoiled* (transient excursions) by a
Fourier Probabilistic Neural Network (FPNN) trained on four handcrafted features.

The repository is fully self-contained: all algorithms are implemented from
scratch in NumPy/SciPy, a suite of 158 unit tests verifies every component, and
two entry-point scripts drive end-to-end experiments.

---

## Repository Structure

```
changepoint-evdecafs/
├── config/
│   └── params.yaml              # Central hyperparameter config
├── data/
│   └── raw/                     # Auto-created; CSV caches land here
├── notebooks/
│   └── tester.ipynb             # 14-cell interactive component tester
├── results/
│   ├── figures/                 # PDF/PNG outputs from pipeline runs
│   └── tables/                  # CSV metrics and sensitivity tables
├── scripts/
│   ├── run_pipeline.py          # Full end-to-end pipeline (603 lines)
│   └── run_phase1_comparison.py # Phase I variant comparison (231 lines)
├── src/
│   ├── data/loader.py           # Bitcoin + well-log data loaders
│   ├── evaluation/
│   │   ├── classification_metrics.py
│   │   ├── hausdorff.py
│   │   ├── mrl_index.py
│   │   └── sensitivity.py
│   ├── phase1/
│   │   ├── ar1_model.py         # Yule-Walker AR(1) estimation
│   │   ├── decafs.py            # DP recursion (Algorithm 1)
│   │   ├── evt_penalty.py       # GPD penalty + exceedance-count penalty
│   │   └── feature_extract.py  # 4-feature extractor (Algorithm 2)
│   ├── phase2/
│   │   ├── fpnn.py              # FourierPNN classifier (Algorithms 4–5)
│   │   ├── labelling.py         # Sustained / recoiled labelling (Algorithm 3)
│   │   ├── smote_balance.py     # SMOTE class balancing
│   │   └── baselines.py        # LR, Isolation Forest, OC-SVM, MLP
│   ├── utils/logging_config.py  # Dual-handler logger (console + file)
│   └── visualization/
│       ├── style.py             # Publication rcParams (LaTeX-optional)
│       ├── run_charts.py        # Time-series run charts
│       ├── roc_curves.py        # Overlaid ROC plots
│       └── sensitivity_heatmap.py
└── tests/                       # 158 pytest tests across 5 files
```

---

## Datasets

### Bitcoin Daily Log-Prices
- **Source**: Yahoo Finance via `yfinance` (BTC-USD, 2014-01-01 → 2024-12-31);
  cached to `data/raw/btc_usd.csv` on first run.
- **Signal**: Natural log of daily closing price.
- **Split**: Training ≤ 2022-12-31, test = 2023-01-01 onwards.
- **Ground truth**: No labelled changepoints; detection is exploratory.

### Well-Log Nuclear Response
- **Source**: Real CSV (`data/raw/welllog.csv`) if present; otherwise a
  synthetic surrogate is generated and cached automatically.
- **Synthetic surrogate** (4 050 observations):
  - 13 segments separated by 12 true changepoints at indices
    {400, 820, 1210, 1320, 1540, 1790, 2050, 2380, 2690, 2990, 3300, 3590}.
  - Segment means drawn uniformly from [70 000, 140 000].
  - AR(1) noise: φ = 0.5, σ_v = 2 000.
  - 20 outlier spikes of magnitude ±30 000 injected at random positions.
- **Split**: 75% training (≈ 3 037 obs), 25% test (≈ 1 013 obs).

---

## Phase I: EV-DeCAFS Changepoint Detection

### Step 1 — AR(1) Parameter Estimation (`src/phase1/ar1_model.py`)

The signal is modelled as:

```
y_t = mu_t + epsilon_t,    epsilon_t = phi * epsilon_{t-1} + v_t
```

Three parameters are estimated from the training series:

| Parameter   | Method                                              |
|-------------|-----------------------------------------------------|
| φ           | Yule-Walker (lag-1 autocorrelation), clipped to (−0.999, 0.999) |
| σ²_v        | Variance of AR(1) residuals `y[1:] − φ·y[:-1]`     |
| σ²_eta      | `max(Var(Δy) − 2σ²_v / (1+|φ|), 1e-8)` — level process variance |

These yield the precision parameters fed to the DP:
`lambda = 1/σ²_eta`,  `gamma = 1/σ²_v`.

### Step 2 — EVI Field Computation (`src/phase1/evt_penalty.py`)

For each time point t, a local window W_t = y[t−w : t+w+1] (half-width w = 50)
is formed. Deviations |y_s − mean(W_t)| are thresholded at their 90th
percentile; a GPD is fitted to the exceedances with fixed location 0:

```
xi_t = shape parameter of GPD fit to exceedances in W_t
```

If fewer than 5 exceedances are present, xi_t = 0. The result is an n-length
**EVI field** capturing local tail-heaviness.

### Step 3 — Adaptive Penalty Schedule

Two penalty variants are implemented:

**GPD-based** (primary, used throughout the paper):
```
alpha_t = alpha_0 * (1 + lambda_ev * max(xi_t, 0))
```
- `alpha_0 = 10`, `lambda_ev = 1.0`
- Penalty is monotone in xi_t; regions with heavy tails get a larger penalty,
  discouraging false changepoints near outlier clusters.

**Exceedance-count** (ablation baseline):
```
alpha_t = alpha_0 * (1 + E_t / (2w+1))
```
- E_t = number of observations in W_t deviating from a preliminary mean
  estimate by more than c·σ_v (c = 2.5).
- Simpler, count-based approximation of the GPD approach.

A **flat penalty** (standard DeCAFS, alpha_t = alpha_0 everywhere) serves as
the non-EVT baseline.

### Step 4 — Dynamic Programming Recursion (`src/phase1/decafs.py`, Algorithm 1)

The cost function to minimise over the level sequence {mu_t} is:

```
Q_t(mu) = min_u { Q_{t-1}(u)
                 + min(lambda * (mu − u)², alpha_t[t])
                 + gamma * ((y[t] − mu) − phi * (y[t-1] − u))² }
```

**Implementation details:**
- The level mu is discretised on a uniform grid of 500 points spanning
  [min(y) − 2σ, max(y) + 2σ].
- At each time step, the (500 × 500) cost matrix is computed by broadcasting:
  `ar1_cost + min(quadratic_penalty, alpha_t[t])`.
- Backtrack pointers (int16, shape n × 500) record the optimal previous grid
  index for each current grid point.
- After the forward pass, the optimal mu sequence is recovered by
  backtracking from the globally minimum final cost.
- **Changepoint criterion**: a changepoint is declared at t if
  `(mu_hat[t] − mu_hat[t−1])² > alpha_t[t] / lambda`.

**Typical runtime**: ~8 s on 3 037 well-log training observations (single core).

---

## Phase I → Phase II Bridge: Feature Extraction & Labelling

### Feature Extraction (`src/phase1/feature_extract.py`, Algorithm 2)

For each detected changepoint τ, two local windows are formed:
- D_minus = y[τ−L : τ]  (pre-change, L = 5)
- D_plus  = y[τ+1 : τ+L+1]  (post-change)

Four features are computed:

| Feature   | Formula                                               | Interpretation              |
|-----------|-------------------------------------------------------|-----------------------------|
| delta_mu  | mean(D_plus) − mean(D_minus)                          | Signed shift magnitude      |
| S         | fraction of D_plus within ε of mean(D_plus)           | Post-change persistence     |
| phi_local | lag-1 autocorrelation of post-change residuals r_t    | Local AR(1) coefficient     |
| V         | var(D_minus) / (var(D_plus) + 1e-10)                  | Pre/post variance ratio     |

The persistence tolerance ε defaults to `median(|y − mu_hat|)` over the
training series.

### Labelling (`src/phase2/labelling.py`, Algorithm 3)

A changepoint is labelled **sustained** (class 1) if and only if:
```
|delta_mu| > kappa_mu   AND   S > kappa_S
```
- `kappa_mu` = 75th percentile of |delta_mu| over all training changepoints.
- `kappa_S = 0.5` (fixed threshold).

Otherwise it is labelled **recoiled** (class 0).

### Class Balancing (`src/phase2/smote_balance.py`)

SMOTE (Synthetic Minority Over-sampling Technique) is applied to the
(feature, label) training pairs. The k-neighbours parameter is automatically
reduced to `min(k_neighbors, n_minority − 1)` when the minority class is
small. If only one class is present, the original data are returned unchanged.

---

## Phase II: Classification

### FPNN — Fourier Probabilistic Neural Network (`src/phase2/fpnn.py`)

The FPNN is a non-parametric Bayesian classifier that estimates
class-conditional densities f(x | c) per feature dimension via truncated
Fourier series with Fejér kernel weighting.

**Training (Algorithm 4):**
1. Scale features to [−0.5, 0.5] with MinMaxScaler.
2. For each class c and harmonic j = 1…J (J = 10):
   ```
   A_{c,d,j} = Σ_{i∈c} cos(π j z_d^(i))
   B_{c,d,j} = Σ_{i∈c} sin(π j z_d^(i))
   ```
3. Apply Fejér weighting: `w_j = (J+1−j) / (N_c · (J+1))`.
4. Stored coefficients: `coef_cos_[c]`, `coef_sin_[c]`, both shape (4, J).

**Prediction (Algorithm 5):**
1. For each test sample and class c, reconstruct the density per feature d:
   ```
   f_d = max(0.5 + Σ_j [A_j cos(πjz_d) + B_j sin(πjz_d)], 1e-10)
   ```
2. Class log-score = Σ_d log(f_d) + log(N / (2 N_c))  (prior correction).
3. Log-sum-exp normalisation → calibrated probabilities.

**Well-log result**: Balanced accuracy = 0.833, MCC = 0.707, AUC-ROC = 1.000
(6 test changepoints, 3 sustained / 3 recoiled).

### Baseline Classifiers (`src/phase2/baselines.py`)

Four baselines are trained and evaluated alongside the FPNN:

| Baseline              | Notes                                                   |
|-----------------------|---------------------------------------------------------|
| Logistic Regression   | L2 regularisation; C searched over {0.01, 0.1, 1, 10} |
| Isolation Forest      | Anomaly detector; contamination = 'auto'                |
| One-Class SVM         | RBF kernel; trained on majority class only              |
| Feedforward NN (MLP)  | Hidden layers [64, 32]; up to 100 epochs                |

Anomaly detectors (+1 / −1 outputs) are remapped to (1 / 0). When ground-truth
labels are unavailable on the test set, auto-labelling using the same
kappa_mu / kappa_S thresholds from training is applied.

---

## Evaluation Metrics

### Classification Metrics (`src/evaluation/classification_metrics.py`)

- **Balanced accuracy** — average of per-class recall (robust to class imbalance).
- **Matthews Correlation Coefficient (MCC)** — single scalar summarising the
  full confusion matrix.
- **AUC-ROC** — area under the ROC curve (requires probability scores).
- Per-class F1, precision, recall.

### MRL Index (`src/evaluation/mrl_index.py`)

The **Mean Run Length** measures detection delay relative to a known changepoint:

```
MRL = T_first − true_cp
```
where T_first is the index of the first detected changepoint at or after
`true_cp`. Detections strictly before `true_cp` are **false positives (FP)**.

The **censored MRL** (used in risk calculations):
- MRL = 0      → ε  (lower bound; avoids division-by-zero)
- MRL = ∞      → T_max  (upper bound; penalises missed detections)
- Otherwise    → MRL

The **censored risk**:
```
R_tilde = (cF * FP) / (cD * MRL_censored)
```
where cF is the cost per false positive and cD the cost per unit delay.

**Well-log result**: EV-DeCAFS (GPD) — 65 total FP, mean MRL = 82 observations.

### Hausdorff Distance (`src/evaluation/hausdorff.py`)

The symmetric Hausdorff distance between the set of detected changepoints and
the set of true changepoints provides a set-theoretic localisation metric:
```
H(A, B) = max( max_{a∈A} min_{b∈B} |a−b|,  max_{b∈B} min_{a∈A} |a−b| )
```
Returns 0 if either set is empty.

**Well-log result**: H(EV-DeCAFS GPD, true CPs) = 2 633 observations.

### Cost-Ratio Sensitivity Analysis (`src/evaluation/sensitivity.py`)

Over a grid of (cF, cD) pairs — cF ∈ {1, 2, 5}, cD ∈ {1, 3, 5, 10} — each
detector is ranked by R_tilde (1 = best). Ties share the minimum rank. Results
are stored as two DataFrames with a MultiIndex (cF, cD):

- **rankings_df**: integer ranks per (cF, cD, detector) triple.
- **raw_df**: corresponding R_tilde float values.

Visualised as two heatmaps: winner-name colour map and winner R_tilde values.

---

## Visualisation (`src/visualization/`)

All figures use a unified publication style (`apply_style()`):
- Serif fonts, 12 pt, 300 DPI.
- LaTeX rendering when `latex` is found on PATH; falls back to mathtext.
- No top/right spines; consistent colour palette.

| Figure                    | Description                                         |
|---------------------------|-----------------------------------------------------|
| `*_train_run_chart`       | Raw series + estimated means + detected CPs (training) |
| `*_test_run_chart`        | Same for the test period                            |
| `*_detector_comparison`   | Multi-panel: GPD vs flat penalty detected CPs       |
| `*_roc`                   | Overlaid ROC curves for all classifiers             |
| `*_sensitivity_ranks`     | Best-detector heatmap over (cF, cD) grid            |
| `*_sensitivity_values`    | Winning R_tilde values over (cF, cD) grid           |

---

## Pipeline Scripts

### `scripts/run_pipeline.py` — Full Pipeline

```
python scripts/run_pipeline.py --dataset [bitcoin|welllog|both]
                               --config config/params.yaml
                               --output-dir results/
                               [--skip-baselines]
```

**Execution flow:**

1. `run_phase1(y_train)` → AR(1) estimation, EVI field, GPD penalty, EV-DeCAFS.
2. `run_phase2_train(y_train, phase1)` → feature extraction, labelling, SMOTE,
   FPNN training.
3. `run_phase2_test(y_test, ...)` → EV-DeCAFS on test, feature extraction,
   auto-labelling, FPNN + baseline evaluation, ROC computation.
4. `run_mrl_analysis(...)` → MRL aggregation, Hausdorff, sensitivity grid.
5. `_make_figures(...)` → all six figure types.
6. Tables saved to `results/tables/`; runtime CSV written.

**Observed runtime (well-log, skip-baselines):**

| Stage             | Time     |
|-------------------|----------|
| AR(1) estimation  | < 0.001 s |
| EVI field (train) | ≈ 120 s  |
| EV-DeCAFS (train) | ≈ 8 s    |
| Feature extraction| < 0.01 s |
| SMOTE + FPNN      | < 0.05 s |
| EVI field (test)  | ≈ 46 s   |
| EV-DeCAFS (test)  | ≈ 2.5 s  |
| **Total**         | ≈ 129 s  |

### `scripts/run_phase1_comparison.py` — Penalty Variant Comparison

```
python scripts/run_phase1_comparison.py --dataset [bitcoin|welllog|both]
```

Fits all three penalty variants (GPD, flat, exceedance-count) on the same
training data, then produces MRL tables, Hausdorff tables, and sensitivity
heatmaps for direct comparison.

---

## Configuration (`config/params.yaml`)

All hyperparameters are centralised; the scripts read the config at startup and
pass values down without hard-coded constants in the source.

| Section     | Key parameters                                                         |
|-------------|------------------------------------------------------------------------|
| `phase1`    | alpha_0=10, lambda_ev=1.0, w=50, q0=0.90, c=2.5                      |
| `labelling` | kappa_mu_percentile=75, kappa_S=0.5, window_L=5                       |
| `fpnn`      | J_harmonics=10, scaling_range=[-0.5, 0.5]                             |
| `smote`     | k_neighbors=5, random_state=42                                         |
| `baselines` | lr_C_range=[0.01,0.1,1,10], fnn_hidden=[64,32]                        |
| `evaluation`| cF_grid=[1,2,5], cD_grid=[1,3,5,10], epsilon=1, Tmax_fraction=0.20   |
| `splitting` | bitcoin_train_end="2022-12-31", welllog_train_fraction=0.75           |
| `visualization` | dpi=300, figure_format="pdf"                                      |

---

## Testing (`tests/`)

158 unit tests across 5 files, run with `pytest tests/ -q`.

| File                         | Scope                                          | Tests |
|------------------------------|------------------------------------------------|-------|
| `test_data.py`               | Loader shape/dtype, cache round-trip, synthetic generation | 18 |
| `test_phase1.py`             | AR(1) properties, EVI field, DP recursion correctness | 28 |
| `test_phase2_prep.py`        | Feature extraction, labelling thresholds, SMOTE | 33 |
| `test_phase2_classifiers.py` | FPNN fit/predict, baselines train/evaluate     | 31 |
| `test_evaluation.py`         | MRL edge cases, Hausdorff, sensitivity, viz smoke tests | 42 |
| `test_smoke.py`              | End-to-end pipeline smoke with tiny synthetic data | 6 |

Notable test cases:
- `test_missed_detection` — MRL = ∞ when no detection at or after true CP.
- `test_perfect_detection` — MRL = 0, FP = 0.
- `test_better_detector_ranks_first` — sensitivity analysis always ranks the
  lower-risk detector at position 1.
- Visualisation cells all use the `Agg` backend to run headlessly in CI.

---

## Notebook (`notebooks/tester.ipynb`)

A 14-cell Jupyter notebook for step-by-step interactive verification:

| Cell | Component tested                                          |
|------|-----------------------------------------------------------|
| 1    | Imports, config loading, inline matplotlib setup          |
| 2    | Data loading — Bitcoin + well-log shape/plot checks       |
| 3    | AR(1) estimation — parameter printout + ACF whiteness plot |
| 4    | EVT penalty — EVI field, GPD vs exceedance-count side-by-side |
| 5    | EV-DeCAFS — run chart with detected CPs and estimated means |
| 6    | Feature extraction — shape, summary stats, histograms     |
| 7    | Labelling — class distribution, kappa_mu check            |
| 8    | SMOTE — before/after scatter plot                         |
| 9    | FPNN — classification report, ROC, coefficient inspection |
| 10   | Baselines — comparison table                              |
| 11   | MRL index — FP / MRL / R_tilde with edge-case assertions  |
| 12   | Sensitivity analysis — heatmap display                    |
| 13   | Hausdorff distance + Spearman correlation with R_tilde    |
| 14   | Full pipeline subprocess dry run + output file verification |

Each cell includes PASS/FAIL assertions and try/except error reporting.

---

## Key Design Decisions

**Grid-based DP instead of FPOP**: The level mu is discretised on a 500-point
uniform grid rather than using a piecewise-quadratic functional representation.
This trades exact optimality for implementation simplicity and is appropriate
for the dataset sizes used in the paper (n ≈ 3 000–4 000).

**int16 backtrack pointers**: Memory footprint of the (n × 500) backtrack array
is halved vs int32 with no loss (500 < 32 767).

**Log-sum-exp in FPNN**: Class log-scores are computed before exponentiation
and normalised via log-sum-exp to prevent underflow when products of small
densities are accumulated across 4 feature dimensions.

**LaTeX-optional style**: `apply_style()` checks `shutil.which("latex")` before
enabling `text.usetex`. LaTeX errors only surface at render time (not at
rcParams assignment), so runtime detection is the only reliable guard.

**Global params in pipeline**: `_make_figures()` accesses configuration via a
module-level `params_global` dict to avoid threading the config object through
all intermediate return values.

---

## Reproducing Results

```bash
# Install dependencies
pip install numpy scipy pandas matplotlib seaborn scikit-learn imbalanced-learn \
            statsmodels tqdm yfinance pyyaml pytest jupyter

# Run well-log experiment (no baselines for speed)
python scripts/run_pipeline.py --dataset welllog --skip-baselines

# Run Phase I variant comparison
python scripts/run_phase1_comparison.py --dataset welllog

# Run full test suite
pytest tests/ -q

# Launch interactive tester
jupyter notebook notebooks/tester.ipynb
```

Output files appear in `results/figures/` (PDFs) and `results/tables/` (CSVs).
