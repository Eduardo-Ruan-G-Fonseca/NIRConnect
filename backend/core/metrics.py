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
    classification_report,
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
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    kappa = float(cohen_kappa_score(y_true, y_pred))
    cm_array = (
        confusion_matrix(y_true, y_pred, labels=labels)
        if labels is not None
        else confusion_matrix(y_true, y_pred)
    )
    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    sensitivity = None
    specificity = None
    sens_by_class = {}
    if labels is None:
        labels = np.unique(y_true)
    row_sums = cm_array.sum(axis=1)
    for idx, label in enumerate(labels):
        sens = cm_array[idx, idx] / row_sums[idx] if row_sums[idx] > 0 else 0.0
        sens_by_class[str(label)] = float(sens)

    if cm_array.shape == (2, 2):
        tn, fp, fn, tp = cm_array.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    desc = (
        "Sensibilidade na PLS-DA: A sensibilidade de acerto na "
        "análise estatística multivariada refere-se à capacidade de um modelo "
        "multivariado em identificar corretamente as relações entre múltiplas "
        "variáveis e fazer previsões precisas. É uma medida de quão bem o "
        "modelo captura a complexidade dos dados e a variabilidade entre as "
        "variáveis. Vários métodos de análise multivariada, como análise de "
        "componentes principais, análise de regressão multivariada e análise "
        "discriminante, são usados para avaliar essa sensibilidade."
    )

    return {
        "Accuracy": float(acc),
        "F1_macro": float(f1_macro),
        "F1_micro": float(f1_micro),
        "Kappa": kappa,
        "ConfusionMatrix": cm_array.tolist(),
        "Sensitivity": None if sensitivity is None else float(sensitivity),
        "Specificity": None if specificity is None else float(specificity),
        "Sensitivity_per_class": sens_by_class,
        "SensitivityDescription": desc,
        "ClassificationReport": report,
        "MacroPrecision": float(prec),
        "MacroRecall": float(rec),
        "MacroF1": float(f1_macro),
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
