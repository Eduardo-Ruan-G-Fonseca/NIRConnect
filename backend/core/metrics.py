import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    explained_variance_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    cohen_kappa_score,
    confusion_matrix,
)

def regression_metrics(y_true, y_pred):
    y_true = y_true.reshape(-1, 1) if y_true.ndim == 1 else y_true
    y_pred = y_pred.reshape(-1, 1) if y_pred.ndim == 1 else y_pred
    # mean_squared_error in some versions of scikit-learn does not support the
    # 'squared' argument. Compute RMSE manually for compatibility.
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = np.mean(np.abs(y_true - y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = float(
        np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-12, None))) * 100
    )
    evs = float(explained_variance_score(y_true, y_pred))
    return {
        "RMSE": float(rmse),
        "MAE": float(mae),
        "R2": float(r2),
        "MAPE": mape,
        "ExplainedVariance": evs,
    }

def classification_metrics(y_true, y_pred, labels=None):
    """Return common classification metrics with safe defaults.

    Unknown predictions (not present in ``labels``) are mapped to the first
    known class to avoid metric errors. ``zero_division=0`` ensures we never
    propagate NaNs.
    """

    classes = np.asarray(labels) if labels is not None else np.unique(y_true)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mask_unknown = ~np.isin(y_pred, classes)
    if mask_unknown.any() and classes.size > 0:
        y_pred = y_pred.copy()
        y_pred[mask_unknown] = classes[0]

    kwargs = dict(labels=classes, zero_division=0)
    acc = float(accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", **kwargs))
    f1_micro = float(f1_score(y_true, y_pred, average="micro", **kwargs))

    if np.unique(y_true).size < 2 or np.unique(y_pred).size < 2:
        kappa = 0.0
    else:
        kappa = float(cohen_kappa_score(y_true, y_pred, labels=classes))

    prec_macro = float(precision_score(y_true, y_pred, average="macro", **kwargs))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", **kwargs))
    cm_array = confusion_matrix(y_true, y_pred, labels=classes)

    return {
        "Accuracy": acc,
        "Kappa": kappa,
        "F1": f1_macro,
        "F1_macro": f1_macro,
        "F1_micro": f1_micro,
        "MacroPrecision": prec_macro,
        "MacroRecall": rec_macro,
        "MacroF1": f1_macro,
        "confusion_matrix": cm_array.tolist(),
        "labels": classes.tolist(),
    }

def vip_scores(pls_model, X, Y):
    T = pls_model.x_scores_
    W = pls_model.x_weights_
    Q = pls_model.y_loadings_
    p, h = W.shape
    S = np.diag(T.T @ T @ Q.T @ Q).reshape(-1, 1)
    total_S = np.sum(S)
    VIP = np.zeros((p,))
    for i in range(p):
        VIP[i] = np.sqrt(p * np.sum((W[i, :]**2) * S.ravel()) / total_S)
    return VIP

def hotelling_t2(pls_model):
    T = pls_model.x_scores_
    cov = np.cov(T, rowvar=False)
    inv_cov = np.linalg.pinv(cov)
    mean_t = np.mean(T, axis=0)
    t2 = [float((ti - mean_t) @ inv_cov @ (ti - mean_t).T) for ti in T]
    return np.array(t2)

def q_residuals(pls_model, X):
    T = pls_model.x_scores_
    P = pls_model.x_loadings_
    Xhat = T @ P.T
    E = X - Xhat
    return np.sum(E**2, axis=1)
