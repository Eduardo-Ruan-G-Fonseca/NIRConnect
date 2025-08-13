from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np
import joblib
import os
from sklearn.model_selection import KFold, cross_validate

from utils.saneamento import saneamento_global
from ml.pipeline import build_pls_pipeline


MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "pls_pipeline.joblib")
MODEL_PATH = os.path.normpath(MODEL_PATH)

router = APIRouter()


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
    X: List[List[float]]
    y: List[float]
    features: Optional[List[str]] = None
    n_components: int = Field(10, ge=1)
    n_splits: int = Field(5, ge=2)


@router.post("/train")
def train(req: TrainRequest):
    X_clean, y_clean, features = saneamento_global(req.X, req.y, req.features)
    if not np.isfinite(X_clean).all():
        raise HTTPException(status_code=400, detail="Dados contêm valores não finitos após saneamento")
    if np.isnan(X_clean).sum() != 0:
        raise HTTPException(status_code=400, detail="Dados contêm NaN após saneamento")
    if X_clean.shape[1] == 0:
        raise HTTPException(status_code=400, detail="Nenhuma coluna válida para treino")

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
    joblib.dump({"pipeline": pipeline, "features": features}, MODEL_PATH)

    return {
        "r2": r2_scores,
        "rmse": rmse_scores,
        "r2_mean": float(np.mean(r2_scores)),
        "rmse_mean": float(np.mean(rmse_scores)),
        "features": features,
    }


class PredictRequest(BaseModel):
    X: List[List[float]]


@router.post("/predict")
def predict(req: PredictRequest):
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=400, detail="Modelo não treinado")
    model_data = joblib.load(MODEL_PATH)
    pipeline = model_data["pipeline"]
    X = np.asarray(req.X, dtype=float)
    preds = pipeline.predict(X).ravel().tolist()
    return {"predictions": preds}
