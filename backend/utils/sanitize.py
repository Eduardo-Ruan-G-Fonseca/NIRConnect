from __future__ import annotations
import numpy as np
from sklearn.impute import SimpleImputer

__all__ = ["sanitize_X", "sanitize_y", "limit_n_components"]

def sanitize_X(X: np.ndarray) -> np.ndarray:
    """Converte para float, troca ±Inf por NaN, remove colunas 100% NaN e imputa média."""
    X = np.asarray(X, dtype=float)
    # ±Inf -> NaN
    X[np.isinf(X)] = np.nan
    # remove colunas inteiras NaN
    keep = ~np.all(np.isnan(X), axis=0)
    if keep.ndim == 0:  # matriz 1D de segurança
        keep = np.array([True])
    X = X[:, keep]
    # imputa média por coluna
    if X.size == 0:
        return X  # evita quebrar; caller decide
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)
    return X

def sanitize_y(y: np.ndarray, task: str) -> np.ndarray:
    """
    task: 'regression' ou 'classification'
    - regressão -> imputa média
    - classificação -> imputa moda (classe mais frequente)
    """
    y = np.asarray(y).reshape(-1, 1).astype(float)
    y[np.isinf(y)] = np.nan
    if task == "classification":
        imp = SimpleImputer(strategy="most_frequent")
    else:
        imp = SimpleImputer(strategy="mean")
    y = imp.fit_transform(y).ravel()
    return y

def limit_n_components(n_components: int, X: np.ndarray) -> int:
    """
    Limita n_components ao rank possível de X:
    min(n_features, n_samples-1)
    """
    if X is None or X.size == 0:
        return max(1, int(n_components))
    n_samples, n_features = X.shape
    hard_max = max(1, min(n_features, n_samples - 1))
    return int(max(1, min(n_components, hard_max)))
