from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
import uvicorn

from slopeforge.config import Config
from slopeforge.eda.report import generate_eda_report
from slopeforge.evaluation.compare import build_leaderboard
from slopeforge.logging_config import generate_run_id, setup_logging
from slopeforge.training.train import persist_run, train_pipeline
from slopeforge.serving.api import create_app

app = typer.Typer(help="SlopeForge CLI")


@app.command()
def eda(
    input: str = typer.Option(..., "--input", help="Input CSV path"),
    target: str = typer.Option(Config().target, "--target", help="Target column"),
    out: str = typer.Option("reports/eda", "--out", help="Output dir"),
):
    run_id = generate_run_id()
    logger = setup_logging(run_id)
    logger.info("Generating EDA report", extra={"run_id": run_id})
    generate_eda_report(input, target, out, drop_cols=Config().drop_cols)
    logger.info("EDA report generated", extra={"run_id": run_id})


@app.command()
def train(
    input: str = typer.Option(
        "data/raw/house_prices.csv", "--input", help="Input CSV path"
    ),
    target: str = typer.Option(Config().target, "--target", help="Target column"),
    model: str = typer.Option("ridge", "--model", help="Model name"),
    test_size: float = typer.Option(Config().test_size, "--test-size"),
    cv_folds: Optional[int] = typer.Option(None, "--cv-folds"),
):
    run_id = generate_run_id()
    logger = setup_logging(run_id)
    logger.info("Loading data", extra={"run_id": run_id})
    df = pd.read_csv(input)

    result = train_pipeline(
        df=df,
        target=target,
        model_name=model,
        drop_cols=Config().drop_cols,
        test_size=test_size,
        random_state=Config().random_state,
        cv_folds=cv_folds,
    )
    payload = persist_run(
        run_dir="models/artifacts",
        metrics_dir="reports/metrics",
        model_name=model,
        pipeline=result["pipeline"],
        metrics=result["metrics"],
        feature_columns=result["feature_columns"],
        split_sizes=result["split_sizes"],
        target=target,
        run_id=run_id,
    )
    logger.info(
        "Training complete",
        extra={"run_id": run_id, "metrics": payload["metrics"]},
    )


@app.command()
def compare(
    runs_dir: str = typer.Option("reports/metrics", "--runs-dir"),
    out: str = typer.Option("reports/metrics/leaderboard.csv", "--out"),
):
    run_id = generate_run_id()
    logger = setup_logging(run_id)
    df = build_leaderboard(runs_dir, out)
    logger.info("Leaderboard saved", extra={"run_id": run_id, "rows": len(df)})


@app.command()
def export(
    best: bool = typer.Option(
        True, "--best/--no-best", help="Export best by RMSE"
    ),
    out: str = typer.Option("models/artifacts/model.joblib", "--out"),
):
    if not best:
        raise typer.BadParameter("Only --best export is supported")
    run_id = generate_run_id()
    logger = setup_logging(run_id)
    runs_dir = Path("reports/metrics")
    run_dicts = []
    for path in runs_dir.glob("run_*.json"):
        run_dicts.append(json.loads(path.read_text(encoding="utf-8")))
    if not run_dicts:
        raise FileNotFoundError(f"No run metrics found in {runs_dir}")
    run_dicts.sort(
        key=lambda r: (r.get("metrics", {}).get("rmse", float("inf")),
                       r.get("metrics", {}).get("mae", float("inf")))
    )
    best_run = run_dicts[0]
    artifact_path = Path(best_run["artifact_path"])
    if not artifact_path.exists():
        raise FileNotFoundError(f"Artifact not found: {artifact_path}")

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(artifact_path.read_bytes())

    metadata = {
        "model_name": best_run.get("model_name"),
        "trained_at": best_run.get("trained_at"),
        "metrics": best_run.get("metrics"),
        "feature_columns": best_run.get("feature_columns"),
    }
    metadata_path = out_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Exported model", extra={"run_id": run_id, "path": str(out_path)})


@app.command()
def serve(
    model: str = typer.Option("models/artifacts/model.joblib", "--model"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
):
    run_id = generate_run_id()
    setup_logging(run_id)
    app_instance = create_app(model)
    uvicorn.run(app_instance, host=host, port=port)


if __name__ == "__main__":
    app()
