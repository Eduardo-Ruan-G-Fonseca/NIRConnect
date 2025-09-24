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


def _validate_sg_params(window: int, poly: int) -> tuple[int, int]:
    if window is None or poly is None:
        raise ValueError("Parâmetros Savitzky-Golay incompletos.")
    if window < 3:
        raise ValueError("window_length deve ser pelo menos 3.")
    if window % 2 == 0:
        raise ValueError("window_length deve ser ímpar.")
    if window <= poly:
        raise ValueError("window_length deve ser maior que polyorder.")
    return int(window), int(poly)


def _safe_apply(func, X, *args, **kwargs):
    try:
        return func(X, *args, **kwargs)
    except Exception:
        return X


def apply_preprocessing(X: np.ndarray, ops: dict) -> np.ndarray:
    """Apply selected preprocessing operations to ``X`` with robust fallbacks."""

    if X is None:
        return None

    Z = np.array(X, dtype=float, copy=True)

    if ops.get("SNV"):
        Z = _safe_apply(snv, Z)
    if ops.get("MSC"):
        Z = _safe_apply(msc, Z)

    if ops.get("SG1"):
        cfg = ops.get("SG1") or {}
        window = int(cfg.get("window", cfg.get("window_length", 11)))
        poly = int(cfg.get("poly", cfg.get("polyorder", 2)))
        try:
            w, p = _validate_sg_params(window, poly)
            Z = _safe_apply(sg1, Z, window=w, poly=p)
        except Exception:
            pass

    if ops.get("SG0"):
        cfg = ops.get("SG0") or {}
        window = int(cfg.get("window", cfg.get("window_length", 11)))
        poly = int(cfg.get("poly", cfg.get("polyorder", 2)))
        try:
            w, p = _validate_sg_params(window, poly)
            Z = _safe_apply(sg0, Z, window=w, poly=p)
        except Exception:
            pass

    Z = np.asarray(Z, dtype=float)
    Z[np.isinf(Z)] = np.nan
    return Z


__all__ = ["snv", "msc", "sg1", "sg0", "apply_preprocessing"]

