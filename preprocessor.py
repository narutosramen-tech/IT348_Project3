from typing import List
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class FraudPreprocessor:
    def __init__(self):
        self.numeric_columns: List[str] = []
        self.categorical_columns: List[str] = []
        self.category_maps = {}

        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, X: pd.DataFrame):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")

        # Identify column types
        self.numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = X.select_dtypes(exclude=[np.number]).columns.tolist()

        # Learn categorical mappings
        for col in self.categorical_columns:
            self.category_maps[col] = {
                category: idx
                for idx, category in enumerate(X[col].dropna().unique())
            }

        # Prepare numeric data for scaler
        X_num = X[self.numeric_columns].copy()
        X_num = X_num.fillna(-999)

        self.scaler.fit(X_num)

        self.fitted = True

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before calling transform")

        X = X.copy()

        # 🔹 Process numeric columns
        if self.numeric_columns:
            X_num = X[self.numeric_columns].copy()
            X_num = X_num.fillna(-999)

            # Scale
            X_scaled = self.scaler.transform(X_num)
            X[self.numeric_columns] = X_scaled

        # 🔹 Process categorical columns
        for col in self.categorical_columns:
            if col in X.columns:
                mapping = self.category_maps.get(col, {})
                X[col] = X[col].map(mapping)
                X[col] = X[col].fillna(-1)

        return X

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.fit(X)
        return self.transform(X)