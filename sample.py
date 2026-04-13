"""
    Author: Jon Bailey
    Written without AI assistance.
"""

import pandas as pd
from dataclasses import dataclass
from typing import Optional

@dataclass
class Sample:
    """
    Represents a dataset split (train/test/etc).
    """
    name: str
    features: pd.DataFrame
    labels: Optional[pd.Series] = None

    def __post_init__(self):
        import warnings

        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(f"name must be a non-empty string, got: {self.name}")

        if not isinstance(self.features, pd.DataFrame):
            raise TypeError("features must be a pandas DataFrame")

        if self.features.empty:
            warnings.warn(f"Sample '{self.name}' has empty features")

        if self.labels is not None:
            if not isinstance(self.labels, pd.Series):
                raise TypeError("labels must be a pandas Series")

            if len(self.features) != len(self.labels):
                raise ValueError("Feature/label length mismatch")

            unique = set(self.labels.unique())
            if not unique.issubset({0, 1}):
                warnings.warn(f"Unexpected label values: {unique}")

    @property
    def has_labels(self) -> bool:
        return self.labels is not None

    @property
    def num_samples(self) -> int:
        return len(self.features)

    @property
    def num_features(self) -> int:
        return self.features.shape[1]
    
    def get_X_y(self):
        if self.labels is None:
            raise ValueError(f"Sample '{self.name}' has no labels.")
        return self.features, self.labels