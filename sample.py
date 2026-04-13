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
    Represents a dataset sample for a specific year.
    """
    year: str
    features: pd.DataFrame
    labels: Optional[pd.Series] = None

    def __post_init__(self):
        """
        Validate the Sample object after initialization.
        """
        import warnings

        # Validate year is not empty
        if not isinstance(self.year, str) or not self.year.strip():
            raise ValueError(f"year must be a non-empty string, got: {self.year}")

        # Validate features is a DataFrame
        if not isinstance(self.features, pd.DataFrame):
            raise TypeError(f"features must be a pandas DataFrame, got: {type(self.features)}")

        # Check for empty features DataFrame
        if self.features.empty:
            warnings.warn(f"Sample for year {self.year} has empty features DataFrame")

        # Validate features dimensions
        if self.features.shape[1] == 0:
            warnings.warn(f"Sample for year {self.year} has 0 features (empty columns)")

        # Validate labels if present
        if self.labels is not None:
            if not isinstance(self.labels, pd.Series):
                raise TypeError(f"labels must be a pandas Series, got: {type(self.labels)}")

            # Check if features and labels have compatible lengths
            if len(self.features) != len(self.labels):
                raise ValueError(
                    f"Mismatched dimensions for year {self.year}: "
                    f"features has {len(self.features)} rows, "
                    f"labels has {len(self.labels)} rows"
                )

            # Check for missing values in labels
            if self.labels.isna().any():
                warnings.warn(f"Sample for year {self.year} has {self.labels.isna().sum()} NaN values in labels")

            # Check label values (assuming binary classification 0/1)
            unique_labels = self.labels.unique()
            if len(unique_labels) == 1:
                warnings.warn(f"Sample for year {self.year} has only one class: {unique_labels[0]}")
            elif not set(unique_labels).issubset({0, 1}):
                warnings.warn(f"Sample for year {self.year} has unexpected label values: {unique_labels}")

    @property
    def has_labels(self) -> bool:
        return self.labels is not None

    @property
    def num_samples(self) -> int:
        return len(self.features)

    @property
    def num_features(self) -> int:
        return self.features.shape[1]