import numpy as np
import math


def snv(X: np.ndarray) -> np.ndarray:
    """Standard normal variate."""
    m = X.mean(axis=1, keepdims=True)
    s = X.std(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return (X - m) / s


def msc(X: np.ndarray) -> np.ndarray:
    """Multiplicative scatter correction."""
    ref = X.mean(axis=0, keepdims=True)
    out = np.empty_like(X, dtype=float)
    for i in range(X.shape[0]):
        b = np.polyfit(ref.ravel(), X[i], 1)
        out[i] = (X[i] - b[1]) / b[0]
    return out


def savgol_1d(X: np.ndarray, window: int = 11, polyorder: int = 2, deriv: int = 1) -> np.ndarray:
    """Aplica Savitzky-Golay ao longo do eixo das variáveis (axis=1).
    Garante parâmetros válidos e corrige uso de factorial."""

    w = int(window)
    p = int(polyorder)
    d = int(deriv)

    if w < 5:
        w = 5
    if w % 2 == 0:
        w += 1
    if p < d:
        p = d
    if w <= p:
        w = p + 1 if (p + 1) % 2 == 1 else p + 2

    try:
        from scipy.signal import savgol_filter
        return savgol_filter(
            X,
            window_length=w,
            polyorder=p,
            deriv=d,
            axis=1,
            mode="interp",
        )
    except Exception:
        half = w // 2
        x = np.arange(-half, half + 1).reshape(-1, 1)
        V = np.hstack([x ** pp for pp in range(p + 1)])
        Vinv = np.linalg.pinv(V)
        e = np.zeros((p + 1, 1))
        e[d, 0] = math.factorial(d)
        coef = (Vinv.T @ e).ravel()
        Xpad = np.pad(X, ((0, 0), (half, half)), mode="reflect")
        out = np.empty_like(X, dtype=float)
        for i in range(X.shape[0]):
            out[i, :] = np.convolve(Xpad[i], coef[::-1], mode="valid")
        return out


def sg_first_derivative(X: np.ndarray, window: int = 11, polyorder: int = 2) -> np.ndarray:
    return savgol_1d(X, window=window, polyorder=polyorder, deriv=1)


def sg_second_derivative(X: np.ndarray, window: int = 11, polyorder: int = 2) -> np.ndarray:
    return savgol_1d(X, window=window, polyorder=polyorder, deriv=2)


def zscore(X: np.ndarray) -> np.ndarray:
    m = X.mean(axis=0, keepdims=True)
    s = X.std(axis=0, keepdims=True)
    s[s == 0] = 1.0
    return (X - m) / s


def minmax_norm(X: np.ndarray) -> np.ndarray:
    mn = X.min(axis=0, keepdims=True)
    mx = X.max(axis=0, keepdims=True)
    denom = np.where(mx - mn == 0, 1.0, mx - mn)
    return (X - mn) / denom


__all__ = [
    "snv",
    "msc",
    "sg_first_derivative",
    "sg_second_derivative",
    "zscore",
    "minmax_norm",
    "savgol_1d",
]
