from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def train_test_split_df(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
