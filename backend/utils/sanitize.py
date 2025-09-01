from __future__ import annotations
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from typing import Optional, Tuple

__all__ = ["sanitize_X", "sanitize_y", "align_X_y", "limit_n_components"]

MISSING_SENTINELS = {"", "na", "n/a", "nan", "null", "none", "-", "–", "—"}


def sanitize_X(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    X[np.isinf(X)] = np.nan
    keep = ~np.all(np.isnan(X), axis=0)
    X = X[:, keep] if keep.ndim == 1 else X
    if X.size == 0:
        return X
    return SimpleImputer(strategy="mean").fit_transform(X)


def _to_float_or_nan(a: np.ndarray) -> np.ndarray:
    out = np.empty(a.shape, dtype=float)
    r = out.ravel(); src = a.ravel()
    for i, v in enumerate(src):
        try: r[i] = float(v)
        except Exception: r[i] = np.nan
    return out


def sanitize_y(y: np.ndarray, task: str) -> Tuple[np.ndarray, Optional[list]]:
    y = np.asarray(y, dtype=object).reshape(-1, 1)
    # normaliza sentinelas de missing (mantém dtype=object para strings)
    for i in range(y.shape[0]):
        v = y[i, 0]
        if v is None: y[i, 0] = np.nan
        elif isinstance(v, str) and v.strip().casefold() in MISSING_SENTINELS:
            y[i, 0] = np.nan

    if task == "classification":
        y_imp = SimpleImputer(strategy="most_frequent").fit_transform(y).ravel().astype(object)
        le = LabelEncoder()
        y_enc = le.fit_transform(y_imp)     # 0..K-1
        return y_enc.astype(int), list(le.classes_)
    else:
        yf = _to_float_or_nan(y.ravel()).reshape(-1, 1)
        yf = SimpleImputer(strategy="mean").fit_transform(yf).ravel()
        return yf, None


def align_X_y(X: np.ndarray, y: np.ndarray):
    X = np.asarray(X); y = np.asarray(y)
    if y.ndim > 1: y = y.ravel()
    n = min(X.shape[0], y.shape[0])
    X = X[:n]; y = y[:n]
    mask = np.ones_like(y, dtype=bool)
    if y.dtype.kind == "f":
        mask = ~np.isnan(y)
    return X[mask], y[mask], mask


def limit_n_components(n_components: int, X: np.ndarray) -> int:
    if X is None or X.size == 0: return int(max(1, n_components))
    n_samples, n_features = X.shape
    hard = max(1, min(n_features, n_samples - 1))
    return int(max(1, min(int(n_components), hard)))

