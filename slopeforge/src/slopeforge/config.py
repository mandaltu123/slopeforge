from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class Config:
    target: str = "SalePrice"
    drop_cols: List[str] = field(default_factory=lambda: ["Id"])
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
