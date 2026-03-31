from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

APP_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = APP_ROOT / "models"

app = FastAPI(title="Olist Marketing Intelligence API")

conversion_model = None
feature_columns = None


class ScoreRequest(BaseModel):
    payload: Dict[str, Any]


@app.on_event("startup")
def startup() -> None:
    global conversion_model, feature_columns
    model_path = MODELS_DIR / "conversion_model.joblib"
    feat_path = MODELS_DIR / "feature_columns.json"
    if model_path.exists() and feat_path.exists():
        conversion_model = joblib.load(model_path)
        feature_columns = json.loads(feat_path.read_text())


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": conversion_model is not None}


@app.post("/predict")
def predict(req: ScoreRequest) -> dict:
    if conversion_model is None or feature_columns is None:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded")

    row = {k: req.payload.get(k) for k in feature_columns}
    df = pd.DataFrame([row])
    prob = float(conversion_model.predict_proba(df)[:, 1][0])
    pred = int(prob >= 0.5)
    return {"conversion_probability": prob, "prediction": pred}
