# SlopeForge

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-regression-green)
![FastAPI](https://img.shields.io/badge/FastAPI-green?logo=fastapi&logoColor=white)
![Typer](https://img.shields.io/badge/Typer-CLI-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

**End-to-end ML regression pipeline with a model leaderboard and FastAPI inference endpoint.** Train multiple regression models on tabular data, compare them on a leaderboard, and serve predictions via REST API or CLI — all in one tool.

## What It Does

- Automated EDA report generation (missing values, distributions, correlations)
- Feature preprocessing: numeric scaling + categorical encoding via scikit-learn ColumnTransformer
- Trains multiple models: Ridge, RandomForest, XGBoost, with hyperparameter tuning
- Builds a leaderboard ranked by RMSE / R² across all runs
- Exports the best model as a `.joblib` artifact
- Serves predictions via FastAPI inference endpoint
- Full Typer CLI (`slopeforge eda`, `slopeforge train`, `slopeforge serve`, `slopeforge leaderboard`)

## Quick Start

```bash
cd SlopeForge-House-Price-Regression-Lab/slopeforge
pip install -r requirements.txt

# Run EDA
python -m slopeforge.cli eda --input data/raw/house_prices.csv

# Train a model
python -m slopeforge.cli train --input data/raw/house_prices.csv --model ridge

# View leaderboard
python -m slopeforge.cli leaderboard

# Serve predictions
python -m slopeforge.cli serve
# API at http://localhost:8000 — POST /predict with JSON features
```

## CLI Commands

| Command | What It Does |
|---|---|
| `eda` | Generate EDA report (HTML) to `reports/eda/` |
| `train` | Train a model, log metrics, save artifact |
| `leaderboard` | Print all runs ranked by RMSE |
| `serve` | Start FastAPI inference server |

## Architecture

```
CSV data
   │
   ▼
EDA Report
   │
   ▼
ColumnTransformer (numeric + categorical)
   │
   ▼
Model Training (Ridge / RF / XGBoost)
   │
   ▼
Leaderboard (RMSE, R², MAE per run)
   │
   ▼
Best model → .joblib artifact → FastAPI /predict
```

## Tech Stack

- **ML**: scikit-learn (pipeline, ColumnTransformer, cross-validation), XGBoost
- **CLI**: Typer
- **API**: FastAPI + Uvicorn
- **EDA**: pandas + automated reporting
- **Artifacts**: joblib

## Project Structure

```
SlopeForge-House-Price-Regression-Lab/
├── slopeforge/
│   ├── src/slopeforge/
│   │   ├── cli.py           # Typer CLI entry point
│   │   ├── data/            # Load, split, schema
│   │   ├── features/        # Preprocessing pipeline
│   │   ├── training/        # Train + hyperparameter tuning
│   │   ├── evaluation/      # Metrics + leaderboard
│   │   ├── eda/             # EDA report generation
│   │   └── serving/         # FastAPI inference API
│   ├── tests/
│   └── requirements.txt
└── README.md
```

## License

MIT — see [LICENSE](LICENSE)
