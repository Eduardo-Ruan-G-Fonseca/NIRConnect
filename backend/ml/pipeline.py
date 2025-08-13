from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.cross_decomposition import PLSRegression
from .transformers import ReplaceInfWithNaN, DropAllNaNColumns


def build_pls_pipeline(n_components: int = 10) -> Pipeline:
    """Build a leak-proof PLS Regression pipeline."""
    return Pipeline(steps=[
        ("inf_to_nan", ReplaceInfWithNaN()),
        ("drop_all_nan_cols", DropAllNaNColumns()),
        ("impute", SimpleImputer(strategy="median")),
        ("var_thresh", VarianceThreshold(0.0)),
        ("pls", PLSRegression(n_components=n_components)),
    ])
