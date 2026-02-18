from __future__ import annotations

from typing import Iterable, List, Tuple

import pandas as pd


def infer_feature_types(
    df: pd.DataFrame,
    target: str,
    drop_cols: Iterable[str] | None = None,
) -> Tuple[List[str], List[str]]:
    drop_cols = set(drop_cols or [])
    columns = [c for c in df.columns if c != target and c not in drop_cols]
    numeric_cols = df[columns].select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in columns if c not in numeric_cols]
    return numeric_cols, categorical_cols
