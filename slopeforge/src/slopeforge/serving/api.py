from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException


def load_metadata(model_path: str | Path) -> dict:
    metadata_path = Path(model_path).with_suffix(".metadata.json")
    if metadata_path.exists():
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    return {}


def build_dataframe(
    payload: Dict[str, Any] | List[Dict[str, Any]],
    feature_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    if isinstance(payload, dict):
        records = [payload]
    else:
        records = payload
    df = pd.DataFrame(records)
    if feature_columns:
        for col in feature_columns:
            if col not in df.columns:
                df[col] = np.nan
        df = df[feature_columns]
    return df


def create_app(model_path: str | Path) -> FastAPI:
    model = joblib.load(model_path)
    metadata = load_metadata(model_path)
    feature_columns = metadata.get("feature_columns")

    app = FastAPI(title="SlopeForge API")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/predict")
    def predict(payload: Union[Dict[str, Any], List[Dict[str, Any]]]):
        try:
            df = build_dataframe(payload, feature_columns)
            preds = model.predict(df)
            return {
                "predictions": [float(p) for p in preds],
                "metadata": {
                    "model_name": metadata.get("model_name"),
                    "trained_at": metadata.get("trained_at"),
                },
            }
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app
