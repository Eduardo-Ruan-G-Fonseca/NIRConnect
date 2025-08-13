from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.model_selection import KFold, cross_validate

from core.io_utils import to_float_matrix, encode_labels_if_needed
from core.saneamento import saneamento_global
from core.bootstrap import train_plsda
try:
    from ml.pipeline import build_pls_pipeline
except Exception:
    from core.ml.pipeline import build_pls_pipeline  # fallback se mover


MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "pls_pipeline.joblib")
MODEL_PATH = os.path.normpath(MODEL_PATH)


# Expose PLS endpoints under a dedicated tag so they appear clearly in /docs
# and group them under the /model prefix
router = APIRouter(prefix="/model", tags=["Model"])

# Make router importable via ``from routers.model import router``
__all__ = ["router"]





class PreprocessRequest(BaseModel):
    X: List[List[float]]
    y: Optional[List[float]] = None
    features: Optional[List[str]] = None
    methods: Optional[List] = None


@router.post("/preprocess")
def preprocess(req: PreprocessRequest):
    X = np.asarray(req.X, dtype=float)
    nan_before = np.isnan(X).sum()
    if req.methods:
        from core.preprocessing import apply_methods
        X = apply_methods(X, req.methods)
    X_clean, y_clean, features = saneamento_global(X, req.y, req.features)
    nan_after = np.isnan(X_clean).sum()
    preview = X_clean[:5].tolist()
    return {
        "shape_before": list(np.asarray(req.X).shape),
        "shape_after": list(X_clean.shape),
        "nans_before": int(nan_before),
        "nans_after": int(nan_after),
        "preview": preview,
        "features": features,
        "y": y_clean.tolist() if y_clean is not None else None,
    }


class TrainRequest(BaseModel):
    X: List[List[Any]]                           # aceita strings
    y: List[Union[float, int, str]]              # aceita rótulos
    features: Optional[List[str]] = None
    n_components: int = Field(10, ge=1)
    n_splits: int = Field(5, ge=2)
    classification: bool = False


@router.post("/train")
def train(req: TrainRequest):
    X = to_float_matrix(req.X)
    if req.classification:
        y_arr, class_mapping, n_classes = encode_labels_if_needed(req.y)
        if n_classes > 2:
            raise HTTPException(
                status_code=400,
                detail=f"PLS-DA atual suporta 2 classes (encontradas {n_classes}).",
            )
    else:
        y_arr = pd.to_numeric(pd.Series(req.y), errors="coerce").to_numpy(dtype=float)
        class_mapping = {}
    X_clean, y_clean, features = saneamento_global(X, y_arr, req.features)
    if (not np.isfinite(X_clean).all()) or np.isnan(X_clean).sum():
        raise HTTPException(status_code=400, detail="Dados inválidos mesmo após saneamento")
    if X_clean.shape[1] == 0:
        raise HTTPException(status_code=400, detail="Nenhuma coluna válida para treino")

    if req.classification:
        model, metrics, extra = train_plsda(X_clean, y_clean, n_components=req.n_components)
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump({"pipeline": model, "features": features, "class_mapping": class_mapping}, MODEL_PATH)
        return {"metrics": metrics, "vip": extra.get("vip"), "features": features, "class_mapping": class_mapping}

    pipeline = build_pls_pipeline(req.n_components)
    cv = KFold(n_splits=req.n_splits, shuffle=True, random_state=42)
    cv_results = cross_validate(
        pipeline,
        X_clean,
        y_clean,
        cv=cv,
        scoring={"r2": "r2", "rmse": "neg_root_mean_squared_error"},
        return_train_score=False,
    )
    r2_scores = cv_results["test_r2"].tolist()
    rmse_scores = (-cv_results["test_rmse"]).tolist()

    pipeline.fit(X_clean, y_clean)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"pipeline": pipeline, "features": features, "class_mapping": class_mapping}, MODEL_PATH)

    return {
        "r2": r2_scores,
        "rmse": rmse_scores,
        "r2_mean": float(np.mean(r2_scores)),
        "rmse_mean": float(np.mean(rmse_scores)),
        "features": features,
        "class_mapping": class_mapping,
    }


class PredictRequest(BaseModel):
    X: List[List[Any]]


@router.post("/predict")
def predict(req: PredictRequest, threshold: float = 0.5):
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=400, detail="Modelo não treinado")
    blob: Dict[str, Any] = joblib.load(MODEL_PATH)
    pipeline = blob["pipeline"]
    class_mapping = blob.get("class_mapping", {})
    X = to_float_matrix(req.X)
    scores = pipeline.predict(X).ravel()
    out: Dict[str, Any] = {"predictions": scores.tolist()}
    if class_mapping:
        idx = (scores >= threshold).astype(int).tolist()
        labels = [class_mapping.get(i, str(i)) for i in idx]
        out["labels_pred"] = labels
        out["class_mapping"] = class_mapping
    return out
