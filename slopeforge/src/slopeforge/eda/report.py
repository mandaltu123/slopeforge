from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


def _missing_values(df: pd.DataFrame, top_k: int = 20) -> pd.Series:
    missing = df.isna().sum().sort_values(ascending=False)
    return missing.head(top_k)


def _numeric_correlations(df: pd.DataFrame, target: str, top_k: int = 20) -> pd.Series:
    numeric_df = df.select_dtypes(include=["number"])
    if target not in numeric_df.columns:
        return pd.Series(dtype=float)
    corr = numeric_df.corr(numeric_only=True)[target].drop(labels=[target])
    return corr.sort_values(key=lambda s: s.abs(), ascending=False).head(top_k)


def generate_eda_report(
    input_path: str | Path,
    target: str,
    out_dir: str | Path,
    drop_cols: Iterable[str] | None = None,
) -> dict:
    df = pd.read_csv(input_path)
    if drop_cols:
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    summary = {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_top_columns": _missing_values(df).to_dict(),
        "target_stats": {},
        "numeric_correlations_top": _numeric_correlations(df, target).to_dict(),
    }

    if target in df.columns:
        target_series = df[target].dropna()
        summary["target_stats"] = {
            "mean": float(target_series.mean()),
            "median": float(target_series.median()),
            "skew": float(target_series.skew()),
        }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / "eda_report.html"
    json_path = out_dir / "summary.json"

    html_sections = [
        "<h1>SlopeForge EDA Report</h1>",
        "<h2>Shape</h2>",
        pd.DataFrame([summary["shape"]]).to_html(index=False),
        "<h2>Dtypes</h2>",
        pd.DataFrame(
            [{"column": k, "dtype": v} for k, v in summary["dtypes"].items()]
        ).to_html(index=False),
        "<h2>Missing Values (Top)</h2>",
        summary_table(summary["missing_top_columns"], "column", "missing_count"),
        "<h2>Target Stats</h2>",
        pd.DataFrame([summary["target_stats"]]).to_html(index=False),
        "<h2>Numeric Correlations with Target (Top)</h2>",
        summary_table(summary["numeric_correlations_top"], "column", "correlation"),
    ]
    html_path.write_text("\n".join(html_sections), encoding="utf-8")
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def summary_table(data: dict, col_name: str, value_name: str) -> str:
    df = pd.DataFrame(
        [{"column": k, value_name: v} for k, v in data.items()]
    )
    if df.empty:
        return "<p>No data</p>"
    df.rename(columns={"column": col_name}, inplace=True)
    return df.to_html(index=False)
