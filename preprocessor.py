"""
Author - Jon Bailey
Written with no assistance from AI. Claude did program and execute a test script for this class.
"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


class FraudPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, verbose=False):
        self.numeric_columns = []
        self.categorical_columns = []
        self.category_maps = {}
        self.id_columns = []
        self.scaler = StandardScaler()
        self.verbose = verbose

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()

        # Get all numeric and categorical columns
        all_numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = X.select_dtypes(exclude=[np.number]).columns.tolist()

        # Exclude ID columns from numeric columns that get scaled
        # ID columns are unique identifiers and should not be normalized
        id_column_patterns = ['TransactionID', 'TransactionId', 'transactionID', 'transactionid', 'ID', 'Id', 'id']

        self.numeric_columns = []
        self.id_columns = []

        for col in all_numeric_columns:
            # Check if column looks like an ID column (case-insensitive)
            is_id_column = any(pattern.lower() in col.lower() for pattern in id_column_patterns)

            if is_id_column:
                self.id_columns.append(col)
                if self.verbose:
                    print(f"Preprocessor: Excluding ID column '{col}' from scaling")
            else:
                self.numeric_columns.append(col)

        # categorical encoding map
        for col in self.categorical_columns:
            self.category_maps[col] = {
                v: i for i, v in enumerate(X[col].dropna().unique())
            }

        # fit scaler (excluding TransactionID)
        if self.numeric_columns:
            X_num = X[self.numeric_columns].fillna(-999)
            self.scaler.fit(X_num)

        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()

        # Scale numeric columns (excluding TransactionID)
        if self.numeric_columns:
            X_num = X[self.numeric_columns].fillna(-999)
            X[self.numeric_columns] = self.scaler.transform(X_num)

        # categorical
        for col in self.categorical_columns:
            if col in X:
                X[col] = X[col].map(self.category_maps[col]).fillna(-1)

        return X