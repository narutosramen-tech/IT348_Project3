#!/usr/bin/env python3
"""

    Author: Jon Bailey
    Claude assisted with edge case identification, and testing.
"""

import pandas as pd
from scipy.stats import ks_2samp
from sample import Sample
from dataset import Dataset

class DriftAnalyzer:
    """
    Performs feature distribution drift analysis across dataset samples.
    """

    def __init__(self, dataset: Dataset, alpha: float = 0.05, mean_threshold: float = 0.001):
        self.dataset = dataset
        self.alpha = alpha
        self.mean_threshold = mean_threshold

    def compare_samples(self, sample_a: Sample, sample_b: Sample, include_skipped_info: bool = False) -> pd.DataFrame:
        """
        Run KS drift test for every feature between two samples.

        Parameters:
        -----------
        sample_a: Sample
            First sample to compare
        sample_b: Sample
            Second sample to compare
        include_skipped_info: bool
            If True, includes columns for skipped features and reasons

        Returns:
        --------
        pd.DataFrame with drift analysis results
        """
        import warnings

        X1 = sample_a.features
        X2 = sample_b.features

        common_features = X1.columns.intersection(X2.columns)

        if len(common_features) == 0:
            warnings.warn(f"No common features between year {sample_a.year} and {sample_b.year}")
            return pd.DataFrame()

        results = []
        skipped_features = 0
        skipped_reasons = {
            'mean_threshold': 0,
            'empty_data': 0,
            'constant_feature': 0,
            'insufficient_data': 0
        }

        for feature in common_features:
            mean_a = X1[feature].mean()
            mean_b = X2[feature].mean()
            mean_diff = abs(mean_a - mean_b)

            # Skip based on mean threshold
            if mean_diff <= self.mean_threshold:
                skipped_features += 1
                skipped_reasons['mean_threshold'] += 1
                continue

            # Get data for KS test
            data_a = X1[feature].dropna()
            data_b = X2[feature].dropna()

            # Handle empty data
            if len(data_a) == 0 or len(data_b) == 0:
                skipped_features += 1
                skipped_reasons['empty_data'] += 1
                continue

            # Handle constant features (zero variance)
            if data_a.nunique() <= 1 and data_b.nunique() <= 1:
                # Both features are constant, check if they have same value
                if len(data_a) > 0 and len(data_b) > 0 and data_a.iloc[0] == data_b.iloc[0]:
                    skipped_features += 1
                    skipped_reasons['constant_feature'] += 1
                    continue
                # If constants are different, we should still test (distributions are different)

            # Check for insufficient data for KS test
            if len(data_a) < 2 or len(data_b) < 2:
                skipped_features += 1
                skipped_reasons['insufficient_data'] += 1
                continue

            try:
                # ks_2samp returns a KstestResult object with statistic and pvalue attributes
                ks_result = ks_2samp(data_a, data_b)
                stat = ks_result.statistic  # type: ignore
                p_value = ks_result.pvalue  # type: ignore
            except Exception as e:
                warnings.warn(f"KS test failed for feature '{feature}' between years "
                             f"{sample_a.year} and {sample_b.year}: {e}")
                skipped_features += 1
                continue

            # Handle None p_value case (shouldn't happen with ks_2samp but being safe)
            drift_detected = p_value < self.alpha  # type: ignore

            result_dict = {
                "feature": feature,
                "year_a": sample_a.year,
                "year_b": sample_b.year,
                "mean_a": mean_a,
                "mean_b": mean_b,
                "mean_diff": mean_diff,
                "ks_statistic": stat,
                "p_value": p_value,
                "drift_detected": drift_detected
            }

            if include_skipped_info:
                result_dict.update({
                    "skipped_features_count": skipped_features,
                    "skipped_mean_threshold": skipped_reasons['mean_threshold'],
                    "skipped_empty_data": skipped_reasons['empty_data'],
                    "skipped_constant_feature": skipped_reasons['constant_feature'],
                    "skipped_insufficient_data": skipped_reasons['insufficient_data']
                })

            results.append(result_dict)

        if include_skipped_info and skipped_features > 0:
            print(f"Skipped {skipped_features} features between years {sample_a.year} and {sample_b.year}:")
            for reason, count in skipped_reasons.items():
                if count > 0:
                    print(f"  - {reason}: {count}")

        return pd.DataFrame(results)
    
    def analyze_consecutive_years(self, include_skipped_info: bool = False) -> pd.DataFrame:
        """
        Run drift analysis for all consecutive samples.

        Parameters:
        -----------
        include_skipped_info: bool
            If True, includes skipped features information in results

        Returns:
        --------
        pd.DataFrame with drift analysis results for consecutive years
        """
        samples = self.dataset.samples
        all_results = []

        for i in range(len(samples) - 1):
            s1 = samples[i]
            s2 = samples[i + 1]

            df = self.compare_samples(s1, s2, include_skipped_info)
            all_results.append(df)

        if all_results:
            return pd.concat(all_results, ignore_index=True)
        return pd.DataFrame()

    def analyze_year_pairs(self, year_pairs: list[tuple[str, str]], include_skipped_info: bool = False) -> pd.DataFrame:
        """
        Run drift analysis for specific year pairs.

        Parameters:
        -----------
        year_pairs: list[tuple[str, str]]
            List of (year_a, year_b) tuples to compare
        include_skipped_info: bool
            If True, includes skipped features information in results

        Returns:
        --------
        pd.DataFrame with drift analysis results for specified year pairs
        """
        import warnings
        from typing import Dict

        # Create dictionary for faster year lookups
        year_to_sample: Dict[str, Sample] = {}
        for sample in self.dataset.samples:
            year_to_sample[sample.year] = sample

        all_results = []

        for year_a, year_b in year_pairs:
            if year_a not in year_to_sample:
                warnings.warn(f"Year {year_a} not found in dataset")
                continue
            if year_b not in year_to_sample:
                warnings.warn(f"Year {year_b} not found in dataset")
                continue

            s1 = year_to_sample[year_a]
            s2 = year_to_sample[year_b]

            df = self.compare_samples(s1, s2, include_skipped_info)
            all_results.append(df)

        if all_results:
            return pd.concat(all_results, ignore_index=True)
        return pd.DataFrame()

    def analyze_all_pairs(self, include_skipped_info: bool = False) -> pd.DataFrame:
        """
        Run drift analysis for all possible year pairs (non-consecutive included).

        Parameters:
        -----------
        include_skipped_info: bool
            If True, includes skipped features information in results

        Returns:
        --------
        pd.DataFrame with drift analysis results for all year pairs
        """
        samples = self.dataset.samples
        years = [sample.year for sample in samples]
        all_results = []

        for i in range(len(years)):
            for j in range(i + 1, len(years)):
                s1 = samples[i]
                s2 = samples[j]
                df = self.compare_samples(s1, s2, include_skipped_info)
                all_results.append(df)

        if all_results:
            return pd.concat(all_results, ignore_index=True)
        return pd.DataFrame()
    
    def drift_summary(self, drift_df: pd.DataFrame, include_skipped: bool = False) -> pd.DataFrame:
        """
        Compute percent of drifting features between year pairs.

        Parameters:
        -----------
        drift_df: pd.DataFrame
            DataFrame returned by compare_samples, analyze_consecutive_years, etc.
        include_skipped: bool
            If True, includes information about skipped features in summary

        Returns:
        --------
        pd.DataFrame with drift analysis summary
        """
        if drift_df.empty:
            return pd.DataFrame()

        # Base summary: drift rate by year pair
        summary = (
            drift_df
            .groupby(["year_a", "year_b"])["drift_detected"]
            .mean()
            .reset_index()
        )

        summary["drift_rate"] = summary["drift_detected"] * 100
        summary = summary.drop(columns=["drift_detected"])

        # Add skipped features information if requested and available
        if include_skipped:
            skipped_cols = [
                'skipped_features_count', 'skipped_mean_threshold',
                'skipped_empty_data', 'skipped_constant_feature',
                'skipped_insufficient_data'
            ]

            # Check which skipped columns are present
            available_skipped_cols = [col for col in skipped_cols if col in drift_df.columns]

            if available_skipped_cols:
                # Get first row for each year pair (skipped counts are the same for all features)
                skipped_summary = (
                    drift_df
                    .drop_duplicates(subset=["year_a", "year_b"])[["year_a", "year_b"] + available_skipped_cols]
                )
                summary = pd.merge(summary, skipped_summary, on=["year_a", "year_b"], how="left")

        return summary
