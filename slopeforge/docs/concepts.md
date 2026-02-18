# Concepts

## What this project does
- Loads a tabular housing dataset and separates features from the target (`SalePrice`).
- Builds a preprocessing + model pipeline, trains it, and evaluates with MAE/RMSE/R2.
- Persists each run’s metrics and the trained pipeline artifact.
- Compares runs to pick the best model and exports a production artifact.
- Serves predictions via a minimal FastAPI service.

## What is the target?
- The **target** is the column we want the model to predict.
- In this project the default target is `SalePrice`, i.e., the final sale price of a house.
- During training, all other columns (after dropping `Id`) are treated as input features.
- At inference time, you send feature values, and the model returns a predicted `SalePrice`.

## What is a feature?
- A **feature** is an input variable used to predict the target.
- Features describe each house (e.g., size, quality, year built).
- In the CSV, features are all columns except the target and any dropped columns (like `Id`).

## What is the median?
- The **median** is the middle value of a sorted list.
- If there are an even number of values, it is the average of the two middle values.
- It is less affected by extreme outliers than the mean (average).
- In this project, the **DummyRegressor** uses the median of `SalePrice` as a simple baseline.

## Key concepts used
- **Supervised regression**: predict a continuous target (`SalePrice`) from features.
- **Train/test split**: we split the dataset into two parts.
  - **Train split (80%)**: the model learns patterns from these rows.
  - **Test split (20%)**: held out during training; used only to evaluate.
  - This simulates performance on unseen data (generalization).
  - In the CLI, this is controlled by `--test-size` (default 0.2).
- **Preprocessing pipeline**:
  - Numeric columns: median imputation + standard scaling.
  - Categorical columns: most-frequent imputation + one-hot encoding.
  - Combined with `ColumnTransformer` to ensure consistent feature handling.
- **Modeling**: multiple regressors are supported so you can compare performance.
  - **DummyRegressor**: a simple baseline that predicts a constant value (median).
    It sets a “minimum bar” for what a real model should beat.
  - **Ridge**: linear regression with L2 regularization to reduce overfitting.
    It works well when the target is roughly a linear combination of features.
  - **RandomForestRegressor**: an ensemble of decision trees averaged together.
    Captures non-linear relationships and interactions with strong performance.
  - **HistGradientBoostingRegressor (HGB)**: gradient-boosted trees built with
    histogram-based splits; fast and accurate for tabular data.
  - **XGBoost (optional)**: a highly optimized gradient boosting library; often
    strong on structured/tabular datasets, but requires the `xgboost` package.
- **Evaluation metrics**: MAE (average absolute error), RMSE (penalizes large errors),
  and R2 (variance explained).
- **Reproducibility**: fixed `random_state`, logged `run_id`, persisted artifacts.

## How we determine the “best” model
- Each training run writes a JSON file under `reports/metrics/` with:
  - model name
  - metrics (MAE, RMSE, R2)
  - artifact path
  - feature columns used
- The `compare` command reads all run JSONs and builds `leaderboard.csv`.
- The leaderboard is sorted by **RMSE ascending**, then **MAE ascending**.
- The `export --best` command picks the top row of the leaderboard (lowest RMSE)
  and copies that run’s model artifact to `models/artifacts/model.joblib`.

## How predictions are produced
- Incoming JSON is converted to a DataFrame.
- Unknown fields are ignored, missing known fields are filled with `NaN`.
- The saved pipeline applies the same preprocessing as training.
- The trained model outputs a numeric prediction for each row.

## How one row becomes a predicted sale price
- A single row is one house described by its feature values (e.g., size, quality).
- The preprocessing step cleans that row (fills missing values, encodes categories).
- The model learned a mapping from features → price during training.
- The trained model applies that mapping to the row and outputs a numeric `SalePrice`.

## Flow diagram (end-to-end)

```mermaid
flowchart TD
    A[Raw CSV: data/raw/house_prices.csv] --> B[EDA Report]
    A --> C[Load & Validate Target]
    C --> D[Drop Id (configurable)]
    D --> E[Split Features/Target]
    E --> F[Train/Test Split (80/20)]
    F --> G[Preprocessing Pipeline]
    G --> H[Train Model]
    H --> I[Evaluate Metrics]
    I --> J[Persist Run JSON + Artifact]
    J --> K[Compare Runs]
    K --> L[Export Best Model]
    L --> M[FastAPI /predict]
```

## Why these metrics?
- **MAE (Mean Absolute Error)**: average absolute error in currency units; easy to interpret.
- **RMSE (Root Mean Squared Error)**: penalizes larger mistakes more strongly; good for ranking models.
- **R2 (R-squared / Coefficient of Determination)**: fraction of variance explained; higher is better, but not always robust.

## What probability theory is used?
- There is no explicit probabilistic model in the default pipeline (no Bayesian model).
- The metrics and validation rely on **statistical estimation**:
  - The test split approximates how the model performs on unseen samples.
  - Metrics are **sample estimates** of true error.
  - Random splits assume data points are **independent and identically distributed (i.i.d.)**.
- If you need **prediction intervals** or uncertainty estimates, you would add:
  - bootstrapping,
  - quantile regression,
  - or Bayesian regression models.

## How to trust the predicted values
- You can never be 100% sure for new data; you can only **estimate accuracy**.
- Use **holdout or cross‑validation** metrics to quantify expected error.
- Compare to a **baseline** (DummyRegressor). A good model should beat it.
- Check **error distributions**: look at large errors, not just averages.
- Verify the input data is **similar to training data** (feature ranges, categories).
## Where the model choice happens
- The CLI does not auto-tune; you explicitly train a model using `--model`.
- You can run multiple models (ridge, rf, hgb, xgb) and compare their metrics.
- The best model is selected by the leaderboard rules above.
