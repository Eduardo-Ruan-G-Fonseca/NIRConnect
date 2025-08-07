import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, r2_score
from .metrics import vip_scores, classification_metrics

from .pls import train_pls


def bootstrap_metrics(X, y, n_components=5, classification=False, n_bootstrap=500, random_state=42):
    rng = np.random.default_rng(random_state)
    metrics_list = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(y), len(y))
        X_bs, y_bs = X[idx], y[idx]
        model, metrics, _ = train_pls(
            X_bs,
            y_bs,
            n_components=n_components,
            classification=classification,
        )
        metrics_list.append(metrics)
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
    pls = PLSRegression(n_components=n_components)
    pls.fit(X, y)
    y_pred = pls.predict(X).ravel()
    r2 = r2_score(y, y_pred)
    rmsep = np.sqrt(mean_squared_error(y, y_pred))

    # cross-validation RMSECV
    n_splits = min(5, len(y)) if len(y) > 1 else 2
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


def train_plsda(
    X: np.ndarray, y: np.ndarray, n_components: int = 5, threshold: float = 0.5
):
    """Train a PLS-DA (classification) model and compute metrics.

    Parameters
    ----------
    X : np.ndarray
        Matrix of predictors.
    y : np.ndarray
        Target labels.
    n_components : int, optional
        Number of latent variables to retain, by default 5.
    threshold : float, optional
        Threshold for binary classification. Only used when the problem is
        binary. Defaults to 0.5.
    """
    y_series = pd.Series(y).astype(str)
    classes = sorted(y_series.unique())
    if len(classes) < 2:
        raise ValueError("A coluna alvo precisa ter pelo menos duas classes distintas.")
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_enc = np.array([class_to_idx[val] for val in y_series])
    if len(classes) > 2:
        Ybin = np.eye(len(classes))[y_enc]
    else:
        Ybin = y_enc.reshape(-1, 1)

    pls = PLSRegression(n_components=n_components)
    pls.fit(X, Ybin)
    Y_pred = pls.predict(X)
    if Y_pred.ndim > 1 and Y_pred.shape[1] > 1:
        idx_pred = np.argmax(Y_pred, axis=1)
    else:
        idx_pred = (Y_pred.ravel() > threshold).astype(int)
    y_pred = np.array([classes[i] for i in idx_pred])

    metrics = classification_metrics(y_series.values, y_pred, labels=classes)
    vip = vip_scores(pls, X, Ybin if Ybin.ndim > 1 else Ybin.reshape(-1, 1)).tolist()
    scores = pls.x_scores_.tolist()
    return pls, metrics, {"vip": vip, "scores": scores, "classes": classes}
