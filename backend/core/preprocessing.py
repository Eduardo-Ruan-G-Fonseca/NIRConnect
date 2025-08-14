import numpy as np
from typing import Iterable, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from scipy.signal import savgol_filter


EPS = 1e-12


def build_spectral_mask(
    wl: np.ndarray,
    spectral_ranges: Optional[Iterable[Tuple[Optional[float], Optional[float]]]],
) -> np.ndarray:
    """Create a boolean mask for ``wl`` based on wavelength intervals.

    Parameters
    ----------
    wl : np.ndarray
        Array of wavelengths.
    spectral_ranges : Optional[Iterable[Tuple[Optional[float], Optional[float]]]]
        Iterable of ``(start, end)`` tuples or dict-like objects with
        ``start`` and ``end`` keys. ``None`` or empty iterables use the
        full range.

    Returns
    -------
    np.ndarray
        Boolean mask with the same shape as ``wl``.
    """

    if spectral_ranges is None:
        return np.ones_like(wl, dtype=bool)

    mask = np.zeros_like(wl, dtype=bool)
    for r in spectral_ranges:
        if r is None:
            continue
        if isinstance(r, dict):
            start = r.get("start")
            end = r.get("end")
        else:
            try:
                start, end = r if len(r) == 2 else (None, None)
            except TypeError:
                start, end = (None, None)

        if start is None or end is None:
            continue
        if isinstance(start, float) and (np.isnan(start) or np.isnan(end)):
            continue

        part = (wl >= start) & (wl <= end)
        mask |= part

    if not mask.any():
        return np.ones_like(wl, dtype=bool)
    return mask


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


def apply_methods(X: np.ndarray, methods: list) -> np.ndarray:
    """Apply preprocessing methods in sequence.

    Parameters
    ----------
    X : np.ndarray
        Matrix of spectra (samples x wavelengths).
    methods : list
        Each element can be either a string with the method name or a
        dictionary in the form ``{"method": "sg1", "params": {...}}``.
    """

    for method in methods:
        if isinstance(method, dict):
            m = str(method.get("method", "")).lower()
            params = method.get("params", {}) or {}
        else:
            m = str(method).lower()
            params = {}

        if m == "snv":
            X = snv(X)
        elif m == "msc":
            X = msc(X)
        elif m == "sg1":
            window = int(params.get("window_length", 11))
            poly = int(params.get("polyorder", 2))
            X = savgol_derivative(X, order=1, window=window, poly=poly)
        elif m == "sg2":
            window = int(params.get("window_length", 11))
            poly = int(params.get("polyorder", 2))
            X = savgol_derivative(X, order=2, window=window, poly=poly)
        elif m == "minmax":
            X = minmax_norm(X)
        elif m == "zscore":
            X = zscore(X)
        elif m == "ncl":
            X = ncl(X)
        elif m == "vn":
            X = vn(X)
    return X


def sanitize_X(X: np.ndarray, feature_names: list[str] | None = None):
    """Sanitize feature matrix ``X``.

    - Cast to float and replace ``±Inf`` with ``NaN``.
    - Drop columns that are entirely ``NaN``.
    - Drop rows that are entirely ``NaN``.
    - Impute remaining ``NaN`` values with the column median.

    Parameters
    ----------
    X : np.ndarray
        Input matrix of shape (samples, features).
    feature_names : list[str] | None, optional
        Optional list of feature names to filter alongside ``X``.

    Returns
    -------
    tuple
        Sanitized ``X`` and corresponding ``feature_names`` (if provided).
    """

    X = np.asarray(X, dtype=float)
    X[~np.isfinite(X)] = np.nan

    # Column filter
    col_ok = ~np.isnan(X).all(axis=0)
    if not col_ok.any():
        raise ValueError(
            "Todas as variáveis espectrais ficaram inválidas após o pré-processamento."
        )
    X = X[:, col_ok]
    if feature_names is not None:
        feature_names = [f for i, f in enumerate(feature_names) if col_ok[i]]

    # Row filter
    row_ok = ~np.isnan(X).all(axis=1)
    X = X[row_ok, :]

    # Impute remaining NaNs with median
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)

    return X, feature_names
