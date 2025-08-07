import os
import sys
import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.pls import train_pls
from core.bootstrap import bootstrap_metrics
from core.optimization import optimize_nir


def generate_regression_data(n_samples=40, n_features=8, noise=0.1, random_state=0):
    rng = np.random.default_rng(random_state)
    X = rng.normal(size=(n_samples, n_features))
    coefs = rng.normal(size=n_features)
    y = X @ coefs + rng.normal(scale=noise, size=n_samples)
    return X, y


def generate_classification_data(n_samples=50, n_features=6, random_state=0):
    rng = np.random.default_rng(random_state)
    X = rng.normal(size=(n_samples, n_features))
    weights = rng.normal(size=(n_features,))
    y = (X @ weights > 0).astype(int)
    return X, y


def generate_multiclass_data(n_samples=60, n_features=5, random_state=0):
    rng = np.random.default_rng(random_state)
    X = rng.normal(size=(n_samples, n_features))
    weights = rng.normal(size=(n_features, 3))
    scores = X @ weights
    y = np.argmax(scores, axis=1)
    return X, y


def test_train_pls_regression():
    X, y = generate_regression_data()
    model, metrics, extra = train_pls(X, y, n_components=3)
    assert not model.classification
    assert set(metrics.keys()) == {"RMSE", "MAE", "R2"}
    assert metrics["R2"] > 0.8
    assert len(extra["vip"]) == X.shape[1]


def test_train_pls_classification():
    X, y = generate_classification_data()
    model, metrics, extra = train_pls(X, y, n_components=2, classification=True)
    assert model.classification
    assert set(metrics.keys()) >= {"Accuracy", "F1_macro", "ConfusionMatrix", "Sensitivity", "Specificity", "Sensitivity_per_class"}
    assert metrics["Accuracy"] >= 0.8
    assert len(extra["vip"]) == X.shape[1]


def test_train_plsda_binary_threshold():
    from core.bootstrap import train_plsda
    X, y = generate_classification_data()
    model, metrics, extra = train_plsda(X, y, n_components=2, threshold=0.4)
    assert metrics["Accuracy"] >= 0.5
    assert extra["classes"] == ["0", "1"]


def test_train_pls_multiclass():
    X, y = generate_multiclass_data()
    model, metrics, extra = train_pls(X, y, n_components=2, classification=True)
    assert model.classification
    assert metrics["ConfusionMatrix"]
    assert len(metrics["ConfusionMatrix"]) == 3
    assert metrics["Accuracy"] >= 0.3
    assert list(model.classes_) == ["0", "1", "2"]


def test_bootstrap_metrics():
    X, y = generate_regression_data(n_samples=30, n_features=5)
    boot = bootstrap_metrics(X, y, n_components=2, n_bootstrap=10)
    for key in ["RMSE", "MAE", "R2"]:
        assert key in boot
        stats = boot[key]
        assert {"mean", "ci95_low", "ci95_high"} <= set(stats.keys())


def test_optimize_nir_no_wl():
    X, y = generate_regression_data(n_samples=25, n_features=6)
    res = optimize_nir(X, y, wl=None, classification=False, n_components_list=[1, 2, 3])
    assert len(res) == 3
    rmses = [r["metrics"]["RMSE"] for r in res]
    assert rmses == sorted(rmses)
    assert all(r["range"] is None for r in res)


def test_optimize_nir_with_wl():
    X, y = generate_regression_data(n_samples=30, n_features=10)
    wl = np.linspace(1100, 2500, X.shape[1])
    res = optimize_nir(X, y, wl=wl, classification=False, n_components_list=[2], n_intervals=4)
    assert len(res) > 0
    scores = [r["score"] for r in res]
    assert scores == sorted(scores)
    for r in res:
        assert r["range"] is not None
        wmin, wmax = r["range"]
        assert wl.min() <= wmin <= wmax <= wl.max()


def test_apply_methods_order():
    from core.preprocessing import snv, minmax_norm, apply_methods
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    expected = minmax_norm(snv(X.copy()))
    result = apply_methods(X.copy(), ["snv", "minmax"])
    assert np.allclose(expected, result)


def test_savgol_params():
    from core.preprocessing import apply_methods
    import numpy as np
    wl = np.linspace(0, 2 * np.pi, 21)
    X = np.vstack([np.sin(wl), np.cos(wl)])
    res_default = apply_methods(X.copy(), ["sg1"])
    res_custom = apply_methods(X.copy(), [{"method": "sg1", "params": {"window_length": 5, "polyorder": 2}}])
    assert res_default.shape == res_custom.shape
    assert not np.allclose(res_default, res_custom)


def test_interpretar_vips():
    from core.interpreter import interpretar_vips
    vips = [1.2, 0.8, 0.5]
    wls = [1414, 1500, 1700]
    res = interpretar_vips(vips, wls, top_n=2)
    assert len(res) == 2
    assert res[0]["vip"] == round(vips[0], 3)
    assert res[0]["grupo"] != "Desconhecido"


def test_gerar_resumo_interpretativo():
    from core.interpreter import gerar_resumo_interpretativo
    dados = [
        {"grupo": "Celulose", "vip": 1.2},
        {"grupo": "Água", "vip": 1.1},
        {"grupo": "Celulose", "vip": 0.9},
        {"grupo": "Lignina", "vip": 0.8},
    ]
    resumo = gerar_resumo_interpretativo(dados)
    assert "Celulose" in resumo and "Água" in resumo


def test_optimize_model_grid():
    from core.optimization import optimize_model_grid
    X, y = generate_regression_data(n_samples=20, n_features=6)
    wl = np.linspace(1100, 1500, X.shape[1])
    res = optimize_model_grid(
        X,
        y,
        wl,
        classification=False,
        preprocess_opts=["none", "snv"],
        n_components_range=range(2, 4),
    )
    assert len(res) > 0
    assert all("RMSECV" in r for r in res)
    assert all("n_components" in r for r in res)


def test_train_pls_validation_options():
    X, y = generate_regression_data(n_samples=20, n_features=5)
    _, _, extra = train_pls(X, y, n_components=2, validation_method="LOO")
    assert extra["validation"]["method"] == "LOO"
    assert "cv_metrics" in extra


def test_optimize_model_grid_holdout():
    from core.optimization import optimize_model_grid
    X, y = generate_regression_data(n_samples=20, n_features=6)
    wl = np.linspace(1100, 1500, X.shape[1])
    res = optimize_model_grid(
        X,
        y,
        wl,
        classification=False,
        preprocess_opts=["none"],
        n_components_range=range(2, 3),
        validation_method="Holdout",
        validation_params={"test_size": 0.3},
    )
    assert res[0]["validation"]["method"] == "Holdout"
    assert "n_components" in res[0]
