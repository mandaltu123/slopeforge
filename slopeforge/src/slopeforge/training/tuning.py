from __future__ import annotations

from typing import Dict


def default_search_space(model_name: str) -> Dict[str, list]:
    model_name = model_name.lower()
    if model_name == "ridge":
        return {"model__alpha": [0.1, 1.0, 10.0]}
    if model_name in {"rf", "randomforest"}:
        return {"model__n_estimators": [200, 400]}
    if model_name in {"hgb", "histgradientboosting"}:
        return {"model__max_depth": [4, 6, 8]}
    return {}
