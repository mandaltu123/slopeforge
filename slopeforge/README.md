# SlopeForge — House Price Regression Lab

End-to-end regression pipeline for predicting house sale price from tabular data.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
```

## CLI Usage

EDA report:

```bash
python -m slopeforge.cli eda --input data/raw/house_prices.csv --target SalePrice --out reports/eda
```

Train models:

```bash
python -m slopeforge.cli train --input data/raw/house_prices.csv --target SalePrice --model ridge
python -m slopeforge.cli train --input data/raw/house_prices.csv --target SalePrice --model rf
python -m slopeforge.cli train --input data/raw/house_prices.csv --target SalePrice --model hgb
```

Compare runs:

```bash
python -m slopeforge.cli compare --runs-dir reports/metrics --out reports/metrics/leaderboard.csv
```

Export best model:

```bash
python -m slopeforge.cli export --best --out models/artifacts/model.joblib
```

Serve predictions:

```bash
python -m slopeforge.cli serve --model models/artifacts/model.joblib
```

## Prediction API

Start server (default `http://127.0.0.1:8000`), then:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"OverallQual": 7, "GrLivArea": 1710}'
```

Batch request:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '[{"OverallQual": 7, "GrLivArea": 1710}, {"OverallQual": 5, "GrLivArea": 1260}]'
```

## Concepts

See `docs/concepts.md` for a detailed explanation of the ML flow and concepts.

## Troubleshooting

- Optional XGBoost: if `xgboost` is not installed, `xgb` will fall back to `HistGradientBoostingRegressor`.
- Missing deps: ensure `pip install -r requirements.txt`.
