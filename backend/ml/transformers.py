import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class ReplaceInfWithNaN(BaseEstimator, TransformerMixin):
    """Transformer that converts arrays to float and replaces infinities with NaN."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        X[~np.isfinite(X)] = np.nan
        return X


class DropAllNaNColumns(BaseEstimator, TransformerMixin):
    """Remove columns that are entirely NaN during fit and reuse the mask on transform."""

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.col_ok_ = ~np.isnan(X).all(axis=0)
        if not self.col_ok_.any():
            raise ValueError("No treino do fold, todas as colunas ficaram NaN.")
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return X[:, self.col_ok_]
