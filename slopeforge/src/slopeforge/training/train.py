from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, Tuple

import joblib
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline

from slopeforge.data.schema import infer_feature_types
from slopeforge.evaluation.metrics import compute_metrics
from slopeforge.features.preprocessing import build_preprocessor


def get_model(model_name: str, random_state: int = 42):
    model_name = model_name.lower()
    if model_name == "dummy":
        return DummyRegressor(strategy="median")
    if model_name == "ridge":
        return Ridge(alpha=1.0)
    if model_name in {"rf", "randomforest"}:
        return RandomForestRegressor(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
        )
    if model_name in {"hgb", "histgradientboosting"}:
        return HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=6,
            max_iter=300,
            random_state=random_state,
        )
    if model_name in {"xgb", "xgboost"}:
        try:
            from xgboost import XGBRegressor

            return XGBRegressor(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="reg:squarederror",
                random_state=random_state,
                n_jobs=-1,
            )
        except Exception:
            return get_model("hgb", random_state=random_state)
    raise ValueError(f"Unknown model '{model_name}'")


def build_pipeline(
    df: pd.DataFrame,
    target: str,
    drop_cols: Iterable[str] | None,
    model_name: str,
    random_state: int,
) -> Tuple[Pipeline, list]:
    numeric_cols, categorical_cols = infer_feature_types(df, target, drop_cols)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    model = get_model(model_name, random_state=random_state)
    pipeline = Pipeline(
        steps=[("preprocess", preprocessor), ("model", model)]
    )
    feature_cols = [c for c in df.columns if c != target and c not in (drop_cols or [])]
    return pipeline, feature_cols


def train_pipeline(
    df: pd.DataFrame,
    target: str,
    model_name: str,
    drop_cols: Iterable[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    cv_folds: int | None = None,
) -> dict:
    df = df.copy()
    if drop_cols:
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found")

    pipeline, feature_cols = build_pipeline(
        df, target, drop_cols=None, model_name=model_name, random_state=random_state
    )

    X = df.drop(columns=[target])
    y = df[target]

    if cv_folds and cv_folds > 1:
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        preds = cross_val_predict(pipeline, X, y, cv=kf)
        pipeline.fit(X, y)
        metrics = compute_metrics(y, preds)
        split_sizes = {"n_train": int(len(X)), "n_test": int(len(X))}
    else:
        from slopeforge.data.split import train_test_split_df

        X_train, X_test, y_train, y_test = train_test_split_df(
            X, y, test_size=test_size, random_state=random_state
        )
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        metrics = compute_metrics(y_test, preds)
        split_sizes = {"n_train": int(len(X_train)), "n_test": int(len(X_test))}

    return {
        "pipeline": pipeline,
        "metrics": metrics,
        "feature_columns": feature_cols,
        "split_sizes": split_sizes,
    }


def persist_run(
    run_dir: str | Path,
    metrics_dir: str | Path,
    model_name: str,
    pipeline: Pipeline,
    metrics: dict,
    feature_columns: list,
    split_sizes: dict,
    target: str,
    run_id: str,
) -> dict:
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(run_dir)
    metrics_dir = Path(metrics_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    artifact_path = run_dir / f"run_{timestamp}_{model_name}.joblib"
    joblib.dump(pipeline, artifact_path)

    payload = {
        "run_id": run_id,
        "model_name": model_name,
        "trained_at": timestamp,
        "metrics": metrics,
        "split_sizes": split_sizes,
        "target": target,
        "feature_columns": feature_columns,
        "artifact_path": str(artifact_path),
    }
    metrics_path = metrics_dir / f"run_{timestamp}_{model_name}.json"
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
