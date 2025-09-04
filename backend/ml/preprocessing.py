import numpy as np


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
    """
    SG 1D rápido (sem scipy): usa np.convolve com coeficientes pré-computados.
    Para estabilidade e simplicidade, refletimos nas bordas.
    """
    half = window // 2
    x = np.arange(-half, half + 1).reshape(-1, 1)
    V = np.hstack([x ** p for p in range(polyorder + 1)])
    Vinv = np.linalg.pinv(V)
    e = np.zeros((polyorder + 1, 1))
    e[deriv, 0] = np.math.factorial(deriv)
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
