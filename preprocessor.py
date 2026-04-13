from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


class FraudPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.numeric_columns = []
        self.categorical_columns = []
        self.category_maps = {}
        self.scaler = StandardScaler()

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()

        self.numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = X.select_dtypes(exclude=[np.number]).columns.tolist()

        # categorical encoding map
        for col in self.categorical_columns:
            self.category_maps[col] = {
                v: i for i, v in enumerate(X[col].dropna().unique())
            }

        # fit scaler
        X_num = X[self.numeric_columns].fillna(-999)
        self.scaler.fit(X_num)

        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()

        # numeric
        if self.numeric_columns:
            X_num = X[self.numeric_columns].fillna(-999)
            X[self.numeric_columns] = self.scaler.transform(X_num)

        # categorical
        for col in self.categorical_columns:
            if col in X:
                X[col] = X[col].map(self.category_maps[col]).fillna(-1)

        return X