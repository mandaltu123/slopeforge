from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pandas as pd


def read_run_metrics(runs_dir: str | Path) -> pd.DataFrame:
    runs_path = Path(runs_dir)
    rows: List[dict] = []
    for path in runs_path.glob("run_*.json"):
        with path.open("r", encoding="utf-8") as f:
            rows.append(json.load(f))
    if not rows:
        return pd.DataFrame()
    return pd.json_normalize(rows)


def build_leaderboard(runs_dir: str | Path, out_path: str | Path) -> pd.DataFrame:
    df = read_run_metrics(runs_dir)
    if df.empty:
        raise FileNotFoundError(f"No run metrics found in {runs_dir}")
    df_sorted = df.sort_values(
        by=["metrics.rmse", "metrics.mae"], ascending=[True, True]
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_sorted.to_csv(out_path, index=False)
    return df_sorted
