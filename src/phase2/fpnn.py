"""Fourier Probabilistic Neural Network (FPNN).

Implements Algorithms 4 (training) and 5 (prediction) from the paper.
The FPNN is a non-parametric density estimator that uses Fourier series
with Fejer kernel weights to model the class-conditional feature densities.
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)


class FourierPNN:
    """Fourier Probabilistic Neural Network classifier.

    Estimates class-conditional densities ``f(x|c)`` for each feature
    dimension independently via truncated Fourier series with Fejér kernel
    weighting, then combines them under a product-of-densities assumption.

    Parameters
    ----------
    J:
        Number of Fourier harmonics (spectral resolution).
    scaling_range:
        ``(min, max)`` target range for MinMax feature scaling before the
        Fourier expansion.

    Attributes
    ----------
    scaler_ : MinMaxScaler
        Fitted feature scaler.
    coef_cos_ : dict[int, np.ndarray]
        Fejér-weighted cosine coefficients, shape ``(n_features, J)`` per class.
    coef_sin_ : dict[int, np.ndarray]
        Fejér-weighted sine coefficients, shape ``(n_features, J)`` per class.
    class_counts_ : dict[int, int]
        Number of training samples per class.
    n_samples_ : int
        Total number of training samples.
    classes_ : np.ndarray
        Sorted unique class labels.
    n_features_in_ : int
        Number of feature dimensions.
    """

    def __init__(
        self,
        J: int = 10,
        scaling_range: tuple[float, float] = (-0.5, 0.5),
    ) -> None:
        self.J = J
        self.scaling_range = scaling_range

    # ------------------------------------------------------------------
    # Fitting (Algorithm 4)
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FourierPNN":
        """Train the FPNN (Algorithm 4).

        Steps:

        1. Scale features to ``scaling_range`` with MinMaxScaler.
        2. For each class ``c`` and each feature dimension ``d``, accumulate
           the Fourier sums:
           ``A_j = Σ cos(π j z_d)``  and  ``B_j = Σ sin(π j z_d)``
           over all class-``c`` samples.
        3. Weight by the Fejér kernel: ``w_j = (J+1−j) / (N_c · (J+1))``.

        Parameters
        ----------
        X:
            Feature matrix, shape ``(n_samples, n_features)``.
        y:
            Class labels, shape ``(n_samples,)``.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        self.n_samples_ = len(y)
        self.n_features_in_ = X.shape[1]

        # Fit and transform features
        self.scaler_ = MinMaxScaler(feature_range=self.scaling_range)
        Z = self.scaler_.fit_transform(X)  # (n_samples, n_features)

        j_idx = np.arange(1, self.J + 1, dtype=float)  # (J,)
        # Fejér kernel weights (without the 1/N_c factor applied later)
        fejer_base = (self.J + 1 - j_idx) / (self.J + 1)  # (J,)

        self.coef_cos_ = {}
        self.coef_sin_ = {}
        self.class_counts_ = {}

        for c in self.classes_:
            mask = y == c
            Z_c = Z[mask]  # (N_c, n_features)
            N_c = len(Z_c)
            self.class_counts_[int(c)] = N_c

            # angles[i, d, j] = π · j · z_d^(i)
            # Efficient: compute (N_c, n_features, J) via broadcasting
            angles = np.pi * Z_c[:, :, np.newaxis] * j_idx  # (N_c, n_features, J)

            # Sum cos/sin over samples → (n_features, J)
            sum_cos = np.sum(np.cos(angles), axis=0)
            sum_sin = np.sum(np.sin(angles), axis=0)

            # Apply Fejér weight + 1/N_c normalisation
            w = fejer_base / N_c  # (J,)
            self.coef_cos_[int(c)] = sum_cos * w  # (n_features, J)
            self.coef_sin_[int(c)] = sum_sin * w  # (n_features, J)

        logger.info(
            "FPNN fit — %d samples, %d features, J=%d, classes=%s, counts=%s",
            self.n_samples_,
            self.n_features_in_,
            self.J,
            self.classes_.tolist(),
            self.class_counts_,
        )
        return self

    # ------------------------------------------------------------------
    # Prediction (Algorithm 5)
    # ------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute class probability estimates (Algorithm 5).

        For each test sample and class ``c``:

        1. Scale features.
        2. Compute the Fourier density estimate for each feature::

               f_d = 0.5 + Σ_j [A_j cos(πjz_d) + B_j sin(πjz_d)]
               f_d = max(f_d, 1e-10)

        3. ``P(c) = Π_d f_d · N / (2 · N_c)``  (class-prior correction).
        4. Normalise so probabilities sum to 1.

        Uses log-probabilities internally for numerical stability.

        Parameters
        ----------
        X:
            Feature matrix, shape ``(n_samples, n_features)``.

        Returns
        -------
        proba : np.ndarray, shape ``(n_samples, 2)``
            Columns ordered by ``self.classes_``: [P(class=0), P(class=1)].
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        Z = self.scaler_.transform(X)  # (n_samples, n_features)
        n_samples = len(Z)
        n_classes = len(self.classes_)

        j_idx = np.arange(1, self.J + 1, dtype=float)  # (J,)

        # angles[s, d, j] = π · j · z_d^(s)
        angles = np.pi * Z[:, :, np.newaxis] * j_idx  # (n_samples, n_features, J)
        cos_vals = np.cos(angles)  # (n_samples, n_features, J)
        sin_vals = np.sin(angles)  # (n_samples, n_features, J)

        log_probs = np.zeros((n_samples, n_classes))

        for ci, c in enumerate(self.classes_):
            c = int(c)
            N_c = self.class_counts_[c]

            # f_d: (n_samples, n_features)
            # einsum 'ndj,dj->nd' : for each (sample, feature) dot over J
            f = (
                0.5
                + np.einsum("ndj,dj->nd", cos_vals, self.coef_cos_[c])
                + np.einsum("ndj,dj->nd", sin_vals, self.coef_sin_[c])
            )
            f = np.maximum(f, 1e-10)

            # Sum log densities over features, add log prior correction
            log_prior = np.log(self.n_samples_ / (2.0 * N_c + 1e-12))
            log_probs[:, ci] = np.sum(np.log(f), axis=1) + log_prior

        # Log-sum-exp normalisation
        log_probs -= log_probs.max(axis=1, keepdims=True)
        probs = np.exp(log_probs)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the predicted class for each sample (argmax of predict_proba).

        Parameters
        ----------
        X:
            Feature matrix, shape ``(n_samples, n_features)``.

        Returns
        -------
        y_pred : np.ndarray of int, shape ``(n_samples,)``
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def get_coefficients(self) -> dict:
        """Return the fitted Fourier coefficients for inspection.

        Returns
        -------
        dict with keys:

        - ``'cos'`` : ``{class_label: array of shape (n_features, J)}``
        - ``'sin'`` : ``{class_label: array of shape (n_features, J)}``
        """
        self._check_fitted()
        return {
            "cos": {c: self.coef_cos_[c].copy() for c in self.coef_cos_},
            "sin": {c: self.coef_sin_[c].copy() for c in self.coef_sin_},
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not hasattr(self, "classes_"):
            raise RuntimeError("FourierPNN must be fitted before calling predict.")
