"""
    Author: Jon Bailey
    Written without AI assistance.
    A class to contain Sample objects created from the data dictionary in the data_preprocessor
"""

from typing import Dict, Tuple, List
import pandas as pd
from sample import Sample

class Dataset:
    """
    Holds a collection of Sample objects created from the year data dictionary.
    """

    def __init__(self, data_dict: Dict[str, Tuple[pd.DataFrame, pd.Series]]):
        self.samples: List[Sample] = []

        for year, (X, y) in data_dict.items():
            sample = Sample(year = year, features = X, labels = y)
            self.samples.append(sample)

        self.samples.sort(key=lambda s: s.year)

    def get_year(self, year:str) -> Sample:
        for sample in self.samples:
            if sample.year == year:
                return sample
            
        raise ValueError(f"Year {year} not found.")
    
    def years(self) -> List[str]:
        return [sample.year for sample in self.samples]
    
    def __len__(self):
        return len(self.samples)
    
    def summary(self):
        print("\n DATASET SUMMARY")
        print("-" * 40)

        for sample in self.samples:
            print(f"Year: {sample.year}")
            print(f"  Samples: {sample.num_samples}")
            print(f"  Features: {sample.num_features}")

            if sample.has_labels:
                benign = (sample.labels == 0).sum()
                malware = (sample.labels == 1).sum()

                print(f"  Benign: {benign}")
                print(f"  Malware: {malware}")

            print()