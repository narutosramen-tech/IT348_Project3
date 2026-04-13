"""
    Author: Jon Bailey
    Written without AI assistance.
    A class to contain Sample objects created from the data dictionary in the data_preprocessor
"""

from typing import Dict, Tuple, List
import pandas as pd
from sample import Sample
from sklearn.model_selection import train_test_split
from models import train_and_evaluate_ensemble
from preprocessor import FraudPreprocessor

class Dataset:
    def __init__(self):
        self.samples: List[Sample] = []
        self.preprocessor = None

    def add_sample(self, sample: Sample):
        self.samples.append(sample)

    def get(self, name: str) -> Sample:
        for s in self.samples:
            if s.name == name:
                return s
        raise ValueError(f"Sample '{name}' not found")

    def summary(self):
        print("\nDATASET SUMMARY")
        print("-" * 40)

        for sample in self.samples:
            print(f"Name: {sample.name}")
            print(f"  Rows: {sample.num_samples}")
            print(f"  Features: {sample.num_features}")

            if sample.has_labels:
                legit = (sample.labels == 0).sum()
                fraud = (sample.labels == 1).sum()

                print(f"  Legit: {legit}")
                print(f"  Fraud: {fraud}")
            print()
    
    def train_test_split(
            self,
            sample_name: str,
            test_size = 0.2,
            random_state = 42
            ):
        sample = self.get(sample_name)

        X, y = sample.get_X_y()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            stratify=y,
            random_state=random_state
        )
        
        if self.preprocessor is None:
            return X_train, X_test, y_train, y_test
        
        X_train_p = self.preprocessor.fit_transform(X_train)
        X_test_p = self.preprocessor.transform(X_test)

        return X_train_p, X_test_p, y_train, y_test
    
    def run_ensemble(self, sample_name: str,
                 voting_type: str = "hard",
                 tie_breaker: str = "malware"):
    
        X_train, X_test, y_train, y_test = self.train_test_split(sample_name)

        return train_and_evaluate_ensemble(
            X_train, X_test, y_train, y_test,
            voting_type=voting_type,
            tie_breaker=tie_breaker
        )
    
    def set_preprocessor(
            self,
            preprocessor: FraudPreprocessor
    ):
        self.preprocessor = preprocessor