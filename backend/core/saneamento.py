import numpy as np
from sklearn.impute import SimpleImputer
from fastapi import HTTPException
from typing import Iterable, Sequence, Tuple, Optional


def sanitize_X(X: Iterable, features: Optional[Sequence[str]] = None) -> Tuple[np.ndarray, Optional[Sequence[str]]]:
    """Basic sanitization for feature matrix ``X``.

    - Cast to float and convert ``±inf`` to ``NaN``
    - Drop columns that are entirely ``NaN`` (also filter ``features``)
    - Impute remaining ``NaN`` values with the column median

    Parameters
    ----------
    X: iterable
        2D data array (samples x features)
    features: sequence of str, optional
        Names of the features. If provided, will be filtered to match the
        remaining columns after sanitization.

    Returns
    -------
    X_saneado: ndarray
        Sanitized numeric matrix with no ``NaN`` values.
    features_filtradas: sequence or None
        Filtered feature names corresponding to the remaining columns.
    """

    X = np.asarray(X, dtype=float)
    X[~np.isfinite(X)] = np.nan
    col_ok = ~np.isnan(X).all(axis=0)
    if not col_ok.any():
        raise ValueError("Todas as variáveis ficaram NaN/Inf após pré-processamento.")
    X = X[:, col_ok]
    if features is not None:
        features = [f for i, f in enumerate(features) if col_ok[i]]
    imp = SimpleImputer(strategy="median")
    X = imp.fit_transform(X)
    return X, features


def saneamento_global(X: Iterable, y: Optional[Iterable] = None,
                      features: Optional[Sequence[str]] = None
                      ) -> Tuple[np.ndarray, Optional[np.ndarray], list[str]]:
    """Sanitize matrix X by removing problematic values.

    Steps:
    - cast to float and replace inf/-inf with NaN
    - drop columns entirely NaN (also filter ``features``)
    - drop rows entirely NaN (also align ``y``)
    - impute remaining NaN values using the median of each column

    Parameters
    ----------
    X: iterable
        2D data array (samples x features)
    y: iterable, optional
        Target values aligned with rows of ``X``
    features: sequence of str, optional
        Names of the features. If provided, its length must match
        the number of columns in ``X``.

    Returns
    -------
    X_saneado: ndarray
        Cleaned numeric matrix with no NaNs or infs.
    y_alinhado: ndarray or None
        ``y`` aligned with the remaining rows. ``None`` if ``y`` was ``None``.
    features_filtradas: list[str]
        Feature names corresponding to the remaining columns.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise HTTPException(status_code=422, detail="X deve ser 2D")

    if features is not None and len(features) != X.shape[1]:
        raise HTTPException(status_code=422, detail="Número de features não corresponde a X")
    features = list(features) if features is not None else [f"f{i}" for i in range(X.shape[1])]

    X[~np.isfinite(X)] = np.nan

    col_ok = ~np.isnan(X).all(axis=0)
    if not col_ok.any():
        raise HTTPException(status_code=400, detail="Todas as variáveis espectrais ficaram inválidas após o saneamento.")
    X = X[:, col_ok]
    features = [f for f, k in zip(features, col_ok) if k]

    row_ok = ~np.isnan(X).all(axis=1)
    if not row_ok.any():
        raise HTTPException(status_code=400, detail="Todas as amostras ficaram inválidas após o saneamento.")
    X = X[row_ok]
    y_aligned = None
    if y is not None:
        y = np.asarray(y, dtype=float)
        y_aligned = y[row_ok]

    imp = SimpleImputer(strategy="median")
    X = imp.fit_transform(X)

    if not np.isfinite(X).all():
        raise HTTPException(status_code=400, detail="Dados não finitos após saneamento")

    return X, y_aligned, features
