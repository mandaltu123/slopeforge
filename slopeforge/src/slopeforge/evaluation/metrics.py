from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}
