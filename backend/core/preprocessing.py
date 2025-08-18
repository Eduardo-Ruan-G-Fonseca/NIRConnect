import numpy as np
from typing import Iterable, Optional, Tuple
from scipy.signal import savgol_filter
from fastapi import HTTPException

EPS = 1e-12


def build_spectral_mask(wl: np.ndarray,
                        spectral_ranges: Optional[Iterable[Tuple[Optional[float], Optional[float]]]] | None) -> np.ndarray:
    """OR de intervalos; se vazio/None => usa tudo. Nunca usa None + None."""
    if not spectral_ranges:
        return np.ones_like(wl, dtype=bool)
    mask = np.zeros_like(wl, dtype=bool)
    for rng in spectral_ranges:
        if rng is None:
            continue
        if isinstance(rng, dict):
            a, b = rng.get("start"), rng.get("end")
        else:
            a, b = rng if isinstance(rng, tuple) and len(rng) >= 2 else (None, None)
        if a is None or b is None:
            continue
        lo, hi = (a, b) if a <= b else (b, a)
        part = (wl >= lo) & (wl <= hi)
        mask |= part
    return mask if mask.any() else np.ones_like(wl, dtype=bool)


def sanitize_X(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """±Inf→NaN, drop colunas 100% NaN, imputação por mediana."""
    X = np.where(np.isinf(X), np.nan, X)
    col_ok = ~(np.all(np.isnan(X), axis=0))
    X = X[:, col_ok]
    if np.isnan(X).any():
        med = np.nanmedian(X, axis=0)
        med = np.where(~np.isfinite(med), 0.0, med)
        ii = np.where(np.isnan(X))
        X[ii] = np.take(med, ii[1])
    return X, col_ok


def sg1(X: np.ndarray, window_length: int = 11, polyorder: int = 2) -> np.ndarray:
    wl = int(window_length or 11)
    po = int(polyorder or 2)
    if wl < (po + 2): wl = po + 3
    if wl % 2 == 0: wl += 1
    return savgol_filter(X, window_length=wl, polyorder=po, deriv=1, axis=1)


def snv(X: np.ndarray) -> np.ndarray:
    """Standard Normal Variate with epsilon to avoid div-by-zero."""
    mean = np.mean(X, axis=1, keepdims=True)
    std = np.std(X, axis=1, keepdims=True)
    std = np.where(std < EPS, 1.0, std)
    return (X - mean) / std


def msc(X: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
    """Multiplicative Scatter Correction (robusto a slope~0)."""
    if reference is None:
        reference = np.mean(X, axis=0)
    corrected = np.zeros_like(X, dtype=float)
    eps = 1e-12
    for i in range(X.shape[0]):
        slope, intercept = np.polyfit(reference, X[i], 1)
        if abs(slope) < eps:
            slope = eps
        corrected[i] = (X[i] - intercept) / slope
    return corrected


def savgol_derivative(X: np.ndarray, order: int = 1, window: int = 11, poly: int = 2) -> np.ndarray:
    """Savitzky-Golay derivative (janela válida)."""
    if window < (poly + 2):
        window = poly + 3
    if window % 2 == 0:
        window += 1
    return savgol_filter(X, window_length=window, polyorder=poly, deriv=order, axis=1)


def minmax_norm(X: np.ndarray) -> np.ndarray:
    """Min-Max normalization with zero-range protection."""
    X = np.asarray(X, dtype=float)
    min_ = np.min(X, axis=0, keepdims=True)
    max_ = np.max(X, axis=0, keepdims=True)
    range_ = np.where((max_ - min_) < EPS, 1.0, max_ - min_)
    return (X - min_) / range_


def zscore(X: np.ndarray) -> np.ndarray:
    """Z-score normalization with safe std."""
    X = np.asarray(X, dtype=float)
    mean = np.mean(X, axis=0, keepdims=True)
    std = np.std(X, axis=0, keepdims=True)
    std = np.where(std < EPS, 1.0, std)
    return (X - mean) / std


def ncl(X: np.ndarray) -> np.ndarray:
    """Normalize spectra to constant length."""
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(norm == 0, 1, norm)


def vn(X: np.ndarray) -> np.ndarray:
    """Vector normalization (unit norm)."""
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(norm == 0, 1, norm)


def apply_methods(
    X: np.ndarray | None, methods: list | None = None, wl: np.ndarray | None = None
) -> np.ndarray:
    """Apply preprocessing methods in sequence.

    The function is defensive against being called with ``X`` equal to
    ``None`` which could happen when the client skips the dataset upload
    step.  Instead of raising a generic ``ValueError`` we return a proper
    HTTP error so the API responds with a meaningful message.
    """

    if X is None:
        raise HTTPException(400, "Nenhum dataset carregado. Faça o upload antes.")

    methods = methods or []
    Xp = X.copy() if hasattr(X, "copy") else X
    for method in methods:
        if isinstance(method, dict):
            m = str(method.get("method", "")).lower()
            params = method.get("params", {}) or {}
        else:
            m = str(method).lower()
            params = {}

        if m == "snv":
            Xp = snv(Xp)
        elif m == "msc":
            Xp = msc(Xp)
        elif m == "sg1":
            Xp = sg1(Xp, window_length=params.get("window_length", 11), polyorder=params.get("polyorder", 2))
        elif m == "sg2":
            Xp = savgol_derivative(Xp, order=2, window=int(params.get("window_length", 11)), poly=int(params.get("polyorder", 2)))
        elif m == "minmax":
            Xp = minmax_norm(Xp)
        elif m == "zscore":
            Xp = zscore(Xp)
        elif m == "ncl":
            Xp = ncl(Xp)
        elif m == "vn":
            Xp = vn(Xp)

    Xp, _ = sanitize_X(Xp)
    return Xp
