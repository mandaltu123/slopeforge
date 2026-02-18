from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd


def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def drop_columns(df: pd.DataFrame, drop_cols: Iterable[str] | None) -> pd.DataFrame:
    if not drop_cols:
        return df
    cols = [c for c in drop_cols if c in df.columns]
    if not cols:
        return df
    return df.drop(columns=cols)


def split_features_target(
    df: pd.DataFrame, target: str
) -> Tuple[pd.DataFrame, pd.Series]:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found")
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def load_dataset(
    path: str | Path, target: str, drop_cols: Iterable[str] | None = None
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    df = load_csv(path)
    df = drop_columns(df, drop_cols)
    X, y = split_features_target(df, target)
    return X, y, list(X.columns)
