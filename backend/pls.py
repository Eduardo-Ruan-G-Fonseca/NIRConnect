from __future__ import annotations

from sklearn.cross_decomposition import PLSRegression
from sklearn.multiclass import OneVsRestClassifier
from typing import Any


def make_pls_reg(n_components: int = 2, **kwargs) -> PLSRegression:
    """Create a PLSRegression model.

    Parameters
    ----------
    n_components: int, default=2
        Number of latent components.
    **kwargs: Any
        Extra keyword arguments passed to ``PLSRegression``.
    """

    return PLSRegression(n_components=n_components, **kwargs)


def make_pls_da(
    n_components: int = 2,
    n_classes: int | None = None,
    **kwargs,
) -> Any:
    """Factory for PLS-DA models.

    When ``n_classes`` is greater than two, a one-vs-rest scheme with
    ``PLSRegression`` as the base estimator is used. Otherwise, a single
    ``PLSRegression`` instance is returned. Additional keyword arguments are
    forwarded to ``PLSRegression``.
    """

    if n_classes is not None and n_classes > 2:
        base = PLSRegression(n_components=n_components, **kwargs)
        return OneVsRestClassifier(base)
    return PLSRegression(n_components=n_components, **kwargs)


__all__ = ["make_pls_reg", "make_pls_da"]

