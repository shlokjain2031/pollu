"""Dwarka: Prototype GTWR engine (lightweight, Python).

This module provides a simple, easy-to-read GTWR-like implementation for
prototyping and testing. It performs localized weighted OLS using a
spatio-temporal Gaussian kernel:

    W_ij = exp( -d_ij^2 / h_s^2 - (t_i - t_j)^2 / h_t^2 )

The implementation is intentionally simple (no spatial indexing, no C++
optimizations). It is suitable for small-to-moderate datasets and for unit
tests. For production or larger datasets, replace this with a C++ core or
use a more optimized library.

API:
    GTWRModel.fit(X, y, coords, times=None, h_s=1.0, h_t=1.0)
    GTWRModel.predict(X_pred, coords_pred, times_pred=None)

"""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from typing import Optional

import numpy as np


def _spatio_temporal_kernel(
    d2: np.ndarray, dt2: Optional[np.ndarray], h_s: float, h_t: Optional[float]
):
    """Compute Gaussian spatio-temporal weights.

    d2: squared spatial distances (n,) or (n_points,) array
    dt2: squared temporal differences (n,) or None
    h_s: spatial bandwidth (>0)
    h_t: temporal bandwidth (>0) or None
    """
    w = np.exp(-d2 / (h_s**2))
    if dt2 is not None and h_t is not None:
        w = w * np.exp(-dt2 / (h_t**2))
    return w


@dataclass
class GTWRModel:
    """A lightweight, prototype GTWR-like model.

    Notes:
    - X is expected to be a 2D numpy array (n_samples, n_features).
    - coords is a (n_samples, 2) array of [x, y] or [lon, lat].
    - times is optional; if provided should be shape (n_samples,) with numeric
      timestamps (e.g., POSIX, ordinal day, or season index).
    """

    h_s: float = 1000.0
    h_t: Optional[float] = None
    fitted_: bool = False
    X_: Optional[np.ndarray] = None
    y_: Optional[np.ndarray] = None
    coords_: Optional[np.ndarray] = None
    times_: Optional[np.ndarray] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        coords: np.ndarray,
        times: Optional[np.ndarray] = None,
        h_s: Optional[float] = None,
        h_t: Optional[float] = None,
    ):
        """Store training data and bandwidths. This prototype performs no
        global optimization at fit time; predictions are made by local
        weighted regressions at each query point.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        coords = np.asarray(coords, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.shape[0] != y.shape[0] or X.shape[0] != coords.shape[0]:
            raise ValueError("X, y and coords must have the same number of rows")

        if times is not None:
            times = np.asarray(times, dtype=float).ravel()
            if times.shape[0] != X.shape[0]:
                raise ValueError("times must have same length as X and y")

        self.X_ = X
        self.y_ = y
        self.coords_ = coords
        self.times_ = times
        if h_s is not None:
            self.h_s = float(h_s)
        if h_t is not None:
            self.h_t = None if math.isnan(h_t) else float(h_t)

        self.fitted_ = True
        return self

    def _squared_spatial_distances(
        self, coords_a: np.ndarray, coords_b: np.ndarray
    ) -> np.ndarray:
        """Return squared Euclidean distances between rows of coords_a and coords_b.

        coords_a: (m,2), coords_b: (n,2) -> returns (m, n)
        """
        a = coords_a[:, None, :]
        b = coords_b[None, :, :]
        d2 = np.sum((a - b) ** 2, axis=2)
        return d2

    def predict(
        self,
        X_pred: np.ndarray,
        coords_pred: np.ndarray,
        times_pred: Optional[np.ndarray] = None,
        return_residuals: bool = False,
    ):
        """Predict at query points using localized weighted OLS.

        Returns:
            preds: (m,) predictions
            residuals (optional): (m,) estimated local residual std (uncertainty)
        """
        if not self.fitted_:
            raise RuntimeError("Model must be fitted before calling predict")

        X_pred = np.asarray(X_pred, dtype=float)
        coords_pred = np.asarray(coords_pred, dtype=float)
        if X_pred.ndim == 1:
            X_pred = X_pred.reshape(-1, 1)

        if X_pred.shape[1] != self.X_.shape[1]:
            raise ValueError("X_pred must have same number of columns as training X")

        if times_pred is not None:
            times_pred = np.asarray(times_pred, dtype=float).ravel()

        m = X_pred.shape[0]
        preds = np.zeros(m, dtype=float)
        uncertainties = np.zeros(m, dtype=float)

        # Precompute squared distances between prediction points and training points
        d2 = self._squared_spatial_distances(coords_pred, self.coords_)  # (m, n)

        for i in range(m):
            d2_i = d2[i]  # (n,)
            dt2_i = None
            if (
                self.times_ is not None
                and times_pred is not None
                and self.h_t is not None
            ):
                dt = times_pred[i] - self.times_
                dt2_i = dt**2

            w = _spatio_temporal_kernel(d2_i, dt2_i, self.h_s, self.h_t)
            # Avoid degeneracy: enforce minimal weight floor
            w = np.maximum(w, 1e-12)

            # Weighted least squares solution: beta = (X^T W X)^-1 X^T W y
            Xw = self.X_ * w[:, None]
            XT_W_X = Xw.T @ self.X_
            try:
                inv = np.linalg.pinv(XT_W_X)
            except np.linalg.LinAlgError:
                inv = np.linalg.pinv(XT_W_X)

            beta = inv @ (Xw.T @ self.y_)
            xq = X_pred[i]
            preds[i] = float(xq @ beta)

            # Local residual/uncertainty: weighted standard deviation of residuals
            y_hat_local = self.X_ @ beta
            resid = self.y_ - y_hat_local
            # weighted variance
            var = (w * resid**2).sum() / (w.sum() + 1e-12)
            uncertainties[i] = math.sqrt(max(var, 0.0))

        if return_residuals:
            return preds, uncertainties
        return preds

    def save(self, path: str):
        """Serialize model to disk (pickle)."""
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @classmethod
    def load(cls, path: str) -> "GTWRModel":
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, GTWRModel):
            raise TypeError("Loaded object is not a GTWRModel")
        return obj


def run_model():
    print("Running Dwarka GTWR prototype (see GTWRModel in module)")
