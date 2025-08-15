import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from .metrics import vip_scores, classification_metrics

from .pls import train_pls
try:
    from ml.pipeline import fit_plsda_multiclass_final
except Exception:  # pragma: no cover - fallback
    from core.ml.pipeline import fit_plsda_multiclass_final


def bootstrap_metrics(X, y, n_components=5, classification=False, n_bootstrap=500, random_state=42):
    rng = np.random.default_rng(random_state)
    metrics_list = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(y), len(y))
        X_bs, y_bs = X[idx], y[idx]
        res = train_pls(
            X_bs,
            y_bs,
            X_bs,
            y_bs,
            n_components=n_components,
            classification=classification,
        )
        metrics_list.append(res["metrics"])
    # Calcular média e intervalo 95%
    keys = metrics_list[0].keys()
    out = {}
    for k in keys:
        vals = [m[k] for m in metrics_list]
        out[k] = {
            "mean": float(np.mean(vals)),
            "ci95_low": float(np.percentile(vals, 2.5)),
            "ci95_high": float(np.percentile(vals, 97.5))
        }
    return out


def train_plsr(X: np.ndarray, y: np.ndarray, n_components: int = 5):
    """Train a PLS regression model and compute metrics."""
    y = y.astype(float)
    n_max = max(1, min(X.shape[1], X.shape[0] - 1))
    n_components = min(n_components, n_max)
    pls = PLSRegression(n_components=n_components)
    pls.fit(X, y)
    y_pred = pls.predict(X).ravel()
    r2 = r2_score(y, y_pred)
    rmsep = np.sqrt(mean_squared_error(y, y_pred))

    # cross-validation RMSECV
    n_splits = max(2, min(5, len(y)))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    preds = []
    obs = []
    for train_idx, test_idx in kf.split(X):
        model_cv = PLSRegression(n_components=n_components)
        model_cv.fit(X[train_idx], y[train_idx])
        preds.extend(model_cv.predict(X[test_idx]).ravel())
        obs.extend(y[test_idx])
    rmsecv = np.sqrt(mean_squared_error(obs, preds))

    vip = vip_scores(pls, X, y.reshape(-1, 1)).tolist()
    scores = pls.x_scores_.tolist()
    metrics = {"R2": float(r2), "RMSECV": float(rmsecv), "RMSEP": float(rmsep)}
    return pls, metrics, {"vip": vip, "scores": scores}


class PLSDAClassifier:
    """Wrapper for PLS + logistic regression pipeline."""

    def __init__(self, pls: PLSRegression, clf):
        self.pls = pls
        self.clf = clf

    def predict(self, X: np.ndarray) -> np.ndarray:
        T = self.pls.transform(X)
        return self.clf.predict(T)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        T = self.pls.transform(X)
        return self.clf.predict_proba(T)


def train_plsda(
    X: np.ndarray, y: np.ndarray, n_components: int = 5
):
    """Train a multi-class PLS-DA model using logistic regression on latent space."""

    y_series = pd.Series(y).astype(str)
    classes = sorted(y_series.unique())
    if len(classes) < 2:
        raise ValueError("A coluna alvo precisa ter pelo menos duas classes distintas.")

    pls, clf = fit_plsda_multiclass_final(X, y_series.values, n_components)
    model = PLSDAClassifier(pls, clf)
    y_pred = model.predict(X)

    le = LabelEncoder()
    y_enc = le.fit_transform(y_series.values)
    metrics = classification_metrics(y_series.values, y_pred, labels=le.classes_)

    vip = vip_scores(pls, X, y_enc.reshape(-1, 1)).tolist()
    scores = pls.x_scores_.tolist()

    return model, metrics, {"vip": vip, "scores": scores, "classes": le.classes_.tolist()}
