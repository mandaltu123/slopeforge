import pandas as pd
import joblib

from slopeforge.training.train import persist_run, train_pipeline


def _make_df():
    return pd.DataFrame(
        {
            "Id": [1, 2, 3, 4, 5],
            "FeatureA": [10, 20, 30, 40, 50],
            "FeatureB": ["a", "b", "a", "b", "a"],
            "SalePrice": [100000, 120000, 130000, 150000, 160000],
        }
    )


def test_smoke_train_and_predict(tmp_path):
    df = _make_df()
    result = train_pipeline(
        df=df,
        target="SalePrice",
        model_name="dummy",
        drop_cols=["Id"],
        test_size=0.2,
        random_state=42,
    )
    payload = persist_run(
        run_dir=tmp_path / "artifacts",
        metrics_dir=tmp_path / "metrics",
        model_name="dummy",
        pipeline=result["pipeline"],
        metrics=result["metrics"],
        feature_columns=result["feature_columns"],
        split_sizes=result["split_sizes"],
        target="SalePrice",
        run_id="test",
    )

    model = joblib.load(payload["artifact_path"])
    preds = model.predict(df.drop(columns=["SalePrice", "Id"]))
    assert len(preds) == len(df)
