# Concepts

## What this project does
- Loads a tabular housing dataset and separates features from the target (`SalePrice`).
- Builds a preprocessing + model pipeline, trains it, and evaluates with MAE/RMSE/R2.
- Persists each run’s metrics and the trained pipeline artifact.
- Compares runs to pick the best model and exports a production artifact.
- Serves predictions via a minimal FastAPI service.

## Key concepts used
- **Supervised regression**: predict a continuous target (`SalePrice`) from features.
- **Train/test split**: default 80/20 holdout to estimate generalization.
- **Preprocessing pipeline**:
  - Numeric columns: median imputation + standard scaling.
  - Categorical columns: most-frequent imputation + one-hot encoding.
  - Combined with `ColumnTransformer` to ensure consistent feature handling.
- **Modeling**: multiple regressors (Dummy, Ridge, RandomForest, HGB, optional XGBoost).
- **Evaluation metrics**: MAE (average absolute error), RMSE (penalizes large errors),
  and R2 (variance explained).
- **Reproducibility**: fixed `random_state`, logged `run_id`, persisted artifacts.

## How predictions are produced
- Incoming JSON is converted to a DataFrame.
- Unknown fields are ignored, missing known fields are filled with `NaN`.
- The saved pipeline applies the same preprocessing as training.
- The trained model outputs a numeric prediction for each row.
