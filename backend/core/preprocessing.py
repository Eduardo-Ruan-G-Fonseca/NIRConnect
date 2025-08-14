import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.signal import savgol_filter


EPS = 1e-12


def snv(X: np.ndarray) -> np.ndarray:
    """Standard Normal Variate with epsilon to avoid div-by-zero."""
    mean = np.mean(X, axis=1, keepdims=True)
    std = np.std(X, axis=1, keepdims=True)
    std = np.where(std < EPS, 1.0, std)
    return (X - mean) / std


def msc(X: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
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
