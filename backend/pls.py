from __future__ import annotations
from typing import Any, Tuple

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.multiclass import OneVsRestClassifier

__all__ = ["make_pls_reg", "make_pls_da", "sanitize_pls_inputs", "cap_n_components"]


def sanitize_pls_inputs(
    X: np.ndarray,
    y: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    """Sanitise feature matrix and optional target prior to model training.

    Steps performed:

    * Cast to ``float`` and replace ``±Inf`` with ``NaN``.
    * Remove columns that are entirely ``NaN`` or exhibit zero variance.
    * Build a mask of valid rows (at least one finite value) and apply it to
      both ``X`` and ``y`` to preserve alignment.

    No imputation is performed here. Remaining ``NaN`` values should be handled
    inside the cross-validation pipeline.
    """

    X_arr = np.asarray(X, dtype=float)
    X_arr[np.isinf(X_arr)] = np.nan

    if X_arr.ndim != 2:
        X_arr = np.atleast_2d(X_arr)

    n_samples, n_features = X_arr.shape
    col_mask = np.ones(n_features, dtype=bool)

    if n_features:
        # remove columns that are entirely NaN
        all_nan_cols = np.all(np.isnan(X_arr), axis=0)
        if np.any(all_nan_cols):
            col_mask[all_nan_cols] = False

        remaining_idx = np.where(col_mask)[0]
        if remaining_idx.size:
            with np.errstate(invalid="ignore"):
                zero_var = np.nanstd(X_arr[:, remaining_idx], axis=0) == 0
            if np.any(zero_var):
                col_mask[remaining_idx[zero_var]] = False
    else:
        col_mask = np.zeros(0, dtype=bool)

    X_arr = X_arr[:, col_mask]

    # valid rows: at least one finite value
    if X_arr.size:
        with np.errstate(invalid="ignore"):
            row_mask = ~np.all(np.isnan(X_arr), axis=1)
    else:
        row_mask = np.zeros(n_samples, dtype=bool)
    X_arr = X_arr[row_mask]

    y_out = None
    if y is not None:
        y_arr = np.asarray(y)
        if y_arr.ndim > 1:
            y_arr = y_arr.reshape(y_arr.shape[0], -1)
        y_out = y_arr[row_mask]
        if y_out.ndim > 1 and y_out.shape[1] == 1:
            y_out = y_out.ravel()
    return X_arr, y_out, row_mask, col_mask


def cap_n_components(n_components: int, X: np.ndarray) -> int:
    if X is None or X.size == 0:
        return int(max(1, n_components))
    n_samples, n_features = X.shape
    hard_limit = max(1, min(int(n_features), max(1, int(n_samples) - 1)))
    return int(max(1, min(int(n_components), hard_limit)))


def make_pls_reg(n_components: int = 2, **kwargs) -> PLSRegression:
    return PLSRegression(n_components=int(n_components), **kwargs)


def make_pls_da(n_components: int = 2, n_classes: int | None = None, **kwargs) -> Any:
    if n_classes is not None and n_classes > 2:
        return OneVsRestClassifier(PLSRegression(n_components=int(n_components), **kwargs))
    return PLSRegression(n_components=int(n_components), **kwargs)

