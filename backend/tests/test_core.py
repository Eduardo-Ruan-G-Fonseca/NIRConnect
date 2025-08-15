import numpy as np
from core.pls import train_pls
from core.bootstrap import bootstrap_metrics
from core.optimization import optimize_nir, optimize_model_grid
from core.preprocessing import apply_methods, snv, minmax_norm


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


def test_train_pls_regression():
    X, y = generate_regression_data()
    res = train_pls(X, y, X, y, n_components=3, classification=False)
    assert "RMSE" in res["metrics"]


def test_train_pls_classification():
    X, y = generate_classification_data()
    res = train_pls(X, y, X, y, n_components=2, classification=True)
    assert "Accuracy" in res["metrics"]


def test_bootstrap_metrics():
    X, y = generate_regression_data(n_samples=30, n_features=5)
    boot = bootstrap_metrics(X, y, n_components=2, n_bootstrap=5)
    assert "RMSE" in boot


def test_optimize_nir():
    X, y = generate_regression_data(n_samples=25, n_features=6)
    res = optimize_nir(X, y, wl=None, classification=False, n_components_list=[1, 2])
    assert len(res) == 2


def test_apply_methods_order():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    expected = minmax_norm(snv(X.copy()))
    result = apply_methods(X.copy(), ["snv", "minmax"])
    assert np.allclose(expected, result)


def test_optimize_model_grid():
    X, y = generate_regression_data(n_samples=20, n_features=6)
    wl = np.linspace(1100, 1500, X.shape[1])
    res = optimize_model_grid(
        X,
        y,
        wl,
        classification=False,
        methods=["none"],
        n_components_range=range(1, 3),
        validation_method="KFold",
        validation_params={"n_splits": 2},
    )
    assert res["results"] and {"id", "prep", "n_components"} <= res["results"][0].keys()
