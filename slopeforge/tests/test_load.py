import pandas as pd

from slopeforge.data.load import load_dataset


def test_load_dataset(tmp_path):
    df = pd.DataFrame(
        {
            "Id": [1, 2],
            "FeatureA": [10, 20],
            "FeatureB": ["x", "y"],
            "SalePrice": [100000, 120000],
        }
    )
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    X, y, cols = load_dataset(csv_path, target="SalePrice", drop_cols=["Id"])

    assert "Id" not in X.columns
    assert len(X) == 2
    assert y.name == "SalePrice"
    assert cols == ["FeatureA", "FeatureB"]
