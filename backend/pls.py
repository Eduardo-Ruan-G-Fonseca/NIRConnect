from __future__ import annotations
from sklearn.cross_decomposition import PLSRegression
from sklearn.multiclass import OneVsRestClassifier
from typing import Any


def make_pls_reg(n_components: int = 2, **kwargs) -> PLSRegression:
    return PLSRegression(n_components=n_components, **kwargs)


def make_pls_da(n_components: int = 2, n_classes: int | None = None, **kwargs) -> Any:
    if n_classes is not None and n_classes > 2:
        return OneVsRestClassifier(PLSRegression(n_components=n_components, **kwargs))
    return PLSRegression(n_components=n_components, **kwargs)

