from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def snv(X: np.ndarray) -> np.ndarray:
    """Standard normal variate."""
    m = X.mean(axis=1, keepdims=True)
    s = X.std(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return (X - m) / s


def msc(X: np.ndarray) -> np.ndarray:
    """Multiplicative scatter correction."""
    ref = X.mean(axis=0, keepdims=True)
    out = np.empty_like(X)
    for i in range(X.shape[0]):
        b = np.polyfit(ref.ravel(), X[i], 1)
        out[i] = (X[i] - b[1]) / b[0]
    return out


def sg1(X: np.ndarray, window: int = 11, poly: int = 2) -> np.ndarray:
    """Savitzky-Golay first derivative."""
    return savgol_filter(X, window_length=window, polyorder=poly, deriv=1, axis=1)


def sg0(X: np.ndarray, window: int = 11, poly: int = 2) -> np.ndarray:
    """Savitzky-Golay smoothing (zero derivative)."""
    return savgol_filter(X, window_length=window, polyorder=poly, deriv=0, axis=1)


def apply_preprocessing(X: np.ndarray, ops: dict) -> np.ndarray:
    """Apply selected preprocessing operations to ``X``.

    ``ops`` is a mapping where keys such as ``"SNV"``, ``"MSC"``, ``"SG1```` and
    ``"SG0"`` enable the respective operations. Values associated with ``SG1`` or
    ``SG0`` may themselves be dictionaries specifying ``window`` and ``poly``
    parameters. The result is always a float ``ndarray`` where infinite values are
    converted to ``NaN``.
    """

    Z = X
    if ops.get("SNV"):
        Z = snv(Z)
    if ops.get("MSC"):
        Z = msc(Z)
    if ops.get("SG1"):
        c = ops["SG1"] or {}
        Z = sg1(Z, window=int(c.get("window", 11)), poly=int(c.get("poly", 2)))
    if ops.get("SG0"):
        c = ops["SG0"] or {}
        Z = sg0(Z, window=int(c.get("window", 11)), poly=int(c.get("poly", 2)))

    Z = np.array(Z, dtype=float)
    Z[np.isinf(Z)] = np.nan
    return Z


__all__ = ["snv", "msc", "sg1", "sg0", "apply_preprocessing"]

