"""
Author - Jon Bailey
Written by Jon Bailey with recommendations from Claude
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any


class EnhancedFeatureEngineer:
    """
    Enhanced feature engineering for fraud detection.

    Adds comprehensive features including time features, transaction features,
    card features, geographic features, email features, and interaction features.
    """

    def __init__(self, feature_config: Optional[Dict[str, bool]] = None, random_state: int = 42):
        """
        Initialize feature engineer with configuration.

        Args:
            feature_config: Dictionary specifying which feature groups to include.
            random_state: Random seed for reproducibility.
        """
        self.feature_config = feature_config or {
            'time_features': True,
            'transaction_features': True,
            'card_features': True,
            'geo_features': True,
            'email_features': True,
            'device_features': True,
            'id_features': True,
            'v_features': True,
            'interaction_features': True,
            'anomaly_features': False,  # Off by default as requires more computation
            'missing_indicators': True,
        }
        self.random_state = random_state
        self.fitted = False

        # Statistics to store during fit
        self.amt_mean = None
        self.amt_std = None
        self.amt_q95 = None
        self.common_email_domains = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'EnhancedFeatureEngineer':
        """
        Compute statistics needed for feature transformations.

        Args:
            X: Training features DataFrame
            y: Target labels (optional, used for target encoding if implemented)

        Returns:
            Self for chaining
        """
        # Transaction amount statistics
        if 'TransactionAmt' in X.columns:
            self.amt_mean = X['TransactionAmt'].mean()
            self.amt_std = X['TransactionAmt'].std()
            self.amt_q95 = X['TransactionAmt'].quantile(0.95)

        # Common email domains
        self.common_email_domains = ['gmail.com', 'yahoo.com', 'hotmail.com',
                                    'outlook.com', 'aol.com', 'live.com']

        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering transformations.

        Args:
            X: Input DataFrame

        Returns:
            DataFrame with engineered features
        """
        X = X.copy()

        # 1. Time-based features
        if self.feature_config.get('time_features') and 'TransactionDT' in X.columns:
            X = self._add_time_features(X)

        # 2. Transaction amount features
        if self.feature_config.get('transaction_features') and 'TransactionAmt' in X.columns:
            X = self._add_transaction_features(X)

        # 3. Card features
        if self.feature_config.get('card_features'):
            X = self._add_card_features(X)

        # 4. Geographic features
        if self.feature_config.get('geo_features'):
            X = self._add_geo_features(X)

        # 5. Email features
        if self.feature_config.get('email_features'):
            X = self._add_email_features(X)

        # 6. Device features (if identity data merged)
        if self.feature_config.get('device_features'):
            X = self._add_device_features(X)

        # 7. ID features (from identity data)
        if self.feature_config.get('id_features'):
            X = self._add_id_features(X)

        # 8. V column features
        if self.feature_config.get('v_features'):
            X = self._add_v_features(X)

        # 9. Interaction features
        if self.feature_config.get('interaction_features'):
            X = self._add_interaction_features(X)

        # 10. Anomaly features
        if self.feature_config.get('anomaly_features'):
            X = self._add_anomaly_features(X)

        # 11. Missing indicators (always last)
        if self.feature_config.get('missing_indicators'):
            X = self._add_missing_indicators(X)

        return X

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    # ===== Individual Feature Engineering Methods =====

    def _add_time_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        # Basic time features
        X['hour'] = (X['TransactionDT'] // 3600) % 24
        X['day'] = (X['TransactionDT'] // 86400)

        # Derived time features
        X['weekday'] = (X['day'] % 7)  # 0=Monday, 6=Sunday
        X['is_weekend'] = X['weekday'].isin([5, 6]).astype(int)

        # Cyclical encoding for hour
        X['hour_sin'] = np.sin(2 * np.pi * X['hour'] / 24)
        X['hour_cos'] = np.cos(2 * np.pi * X['hour'] / 24)

        # Time of day categories
        X['is_night'] = ((X['hour'] >= 0) & (X['hour'] <= 6)).astype(int)
        X['is_morning'] = ((X['hour'] >= 7) & (X['hour'] <= 12)).astype(int)
        X['is_afternoon'] = ((X['hour'] >= 13) & (X['hour'] <= 18)).astype(int)
        X['is_evening'] = ((X['hour'] >= 19) & (X['hour'] <= 23)).astype(int)

        # Seconds within hour
        X['seconds_in_hour'] = X['TransactionDT'] % 3600

        return X

    def _add_transaction_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add transaction amount features."""
        # Log transformation
        X['TransactionAmt_log'] = np.log1p(X['TransactionAmt'])

        # Statistical transformations
        if self.amt_mean is not None and self.amt_std is not None:
            X['TransactionAmt_zscore'] = (X['TransactionAmt'] - self.amt_mean) / self.amt_std

        # Percentile-based features
        X['TransactionAmt_percentile'] = X['TransactionAmt'].rank(pct=True)

        # Threshold-based features
        if self.amt_q95 is not None:
            X['is_high_amount'] = (X['TransactionAmt'] > self.amt_q95).astype(int)

        # Other transformations
        X['TransactionAmt_sqrt'] = np.sqrt(X['TransactionAmt'])
        X['TransactionAmt_reciprocal'] = 1 / (X['TransactionAmt'] + 1e-6)  # Avoid division by zero

        # Binning
        X['TransactionAmt_bin'] = pd.qcut(X['TransactionAmt'], q=5, labels=False, duplicates='drop')

        return X

    def _add_card_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add card-related features."""
        # Card number features
        if 'card1' in X.columns:
            X['card1_first4'] = X['card1'].astype(str).str[:4].fillna('0000')
            X['card1_first6'] = X['card1'].astype(str).str[:6].fillna('000000')

        # Card type features
        if 'card4' in X.columns:
            X['is_visa'] = (X['card4'] == 'visa').astype(int)
            X['is_mastercard'] = (X['card4'] == 'mastercard').astype(int)
            X['is_amex'] = (X['card4'] == 'american express').astype(int)
            X['is_discover'] = (X['card4'] == 'discover').astype(int)

        # Card category features
        if 'card6' in X.columns:
            X['is_debit'] = (X['card6'] == 'debit').astype(int)
            X['is_credit'] = (X['card6'] == 'credit').astype(int)
            X['is_charge'] = (X['card6'] == 'charge').astype(int)

        # Card interaction features
        if all(col in X.columns for col in ['card2', 'card3']):
            X['card2_card3_diff'] = X['card2'] - X['card3']
            X['card2_card3_sum'] = X['card2'] + X['card3']

        # Missing card features
        card_cols = [col for col in X.columns if col.startswith('card')]
        if card_cols:
            X['card_missing_count'] = X[card_cols].isna().sum(axis=1)

        return X

    def _add_geo_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add geographic features."""
        # Address features
        if 'addr1' in X.columns:
            X['addr1_missing'] = X['addr1'].isna().astype(int)

        if 'addr2' in X.columns:
            X['addr2_missing'] = X['addr2'].isna().astype(int)

        if all(col in X.columns for col in ['addr1', 'addr2']):
            X['both_addr_missing'] = (X['addr1_missing'] & X['addr2_missing']).astype(int)

        # Distance features
        if 'dist1' in X.columns and 'dist2' in X.columns:
            X['dist_abs_diff'] = abs(X['dist1'] - X['dist2'])
            X['dist_sum'] = X['dist1'] + X['dist2']

            # Avoid division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                X['dist_ratio'] = X['dist1'] / (X['dist2'].replace(0, np.nan))
            X['dist_ratio'] = X['dist_ratio'].fillna(0)

            X['is_same_location'] = (X['dist1'] == X['dist2']).astype(int)

        return X

    def _add_email_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add email-related features."""
        # Purchaser email features
        if 'P_emaildomain' in X.columns:
            X['P_email_missing'] = X['P_emaildomain'].isna().astype(int)
            if self.common_email_domains is not None:
                X['P_email_is_common'] = X['P_emaildomain'].isin(self.common_email_domains).astype(int)

            # Extract domain parts
            if X['P_emaildomain'].dtype == 'object':
                X['P_email_tld'] = X['P_emaildomain'].str.extract(r'\.([^.]+)$')
                X['P_email_is_free'] = X['P_emaildomain'].str.contains(
                    'gmail|yahoo|hotmail|outlook|aol|live', case=False
                ).astype(int)

        # Recipient email features
        if 'R_emaildomain' in X.columns:
            X['R_email_missing'] = X['R_emaildomain'].isna().astype(int)
            if self.common_email_domains is not None:
                X['R_email_is_common'] = X['R_emaildomain'].isin(self.common_email_domains).astype(int)

        # Email comparison features
        if all(col in X.columns for col in ['P_emaildomain', 'R_emaildomain']):
            X['email_match'] = (X['P_emaildomain'] == X['R_emaildomain']).astype(int)
            X['email_both_missing'] = (X['P_email_missing'] & X['R_email_missing']).astype(int)

        return X

    def _add_device_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add device-related features (requires identity data)."""
        if 'DeviceType' in X.columns:
            X['is_mobile'] = (X['DeviceType'] == 'mobile').astype(int)
            X['is_desktop'] = (X['DeviceType'] == 'desktop').astype(int)

        if 'DeviceInfo' in X.columns and X['DeviceInfo'].dtype == 'object':
            # Device OS features
            X['device_is_android'] = X['DeviceInfo'].str.contains('Android', case=False).astype(int)
            X['device_is_ios'] = X['DeviceInfo'].str.contains('iOS|iPhone', case=False).astype(int)
            X['device_is_windows'] = X['DeviceInfo'].str.contains('Windows', case=False).astype(int)
            X['device_is_mac'] = X['DeviceInfo'].str.contains('Mac|macOS', case=False).astype(int)

            # Device brand features
            X['device_is_samsung'] = X['DeviceInfo'].str.contains('samsung|SM-', case=False).astype(int)
            X['device_is_apple'] = X['DeviceInfo'].str.contains('iPhone|iPad|iOS', case=False).astype(int)

        return X

    def _add_id_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add features from identity columns."""
        id_cols = [col for col in X.columns if col.startswith('id_')]
        if id_cols:
            # Missing value statistics
            X['id_missing_count'] = X[id_cols].isna().sum(axis=1)
            X['id_filled_ratio'] = 1 - (X['id_missing_count'] / len(id_cols))

            # Numeric ID statistics
            numeric_id_cols = [col for col in id_cols if pd.api.types.is_numeric_dtype(X[col])]
            if numeric_id_cols:
                X['id_numeric_mean'] = X[numeric_id_cols].mean(axis=1, skipna=True)
                X['id_numeric_std'] = X[numeric_id_cols].std(axis=1, skipna=True)
                X['id_numeric_sum'] = X[numeric_id_cols].sum(axis=1, skipna=True)

        return X

    def _add_v_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add features from V columns."""
        v_cols = [col for col in X.columns if col.startswith('V')]
        if v_cols:
            # Basic statistics
            X['V_mean'] = X[v_cols].mean(axis=1, skipna=True)
            X['V_std'] = X[v_cols].std(axis=1, skipna=True)
            X['V_sum'] = X[v_cols].sum(axis=1, skipna=True)
            X['V_min'] = X[v_cols].min(axis=1, skipna=True)
            X['V_max'] = X[v_cols].max(axis=1, skipna=True)

            # Missing value statistics
            X['V_missing_count'] = X[v_cols].isna().sum(axis=1)
            X['V_missing_ratio'] = X['V_missing_count'] / len(v_cols)

            # Non-zero statistics
            X['V_non_zero_count'] = (X[v_cols] != 0).sum(axis=1)
            X['V_non_zero_ratio'] = X['V_non_zero_count'] / len(v_cols)

        return X

    def _add_interaction_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add interaction features between different feature groups."""
        # Amount × Time interactions
        if 'TransactionAmt' in X.columns and 'hour' in X.columns:
            X['amt_hour_interaction'] = X['TransactionAmt'] * X['hour']
            X['amt_weekend_interaction'] = X['TransactionAmt'] * X.get('is_weekend', 0)

        # Amount × Card interactions
        if 'TransactionAmt' in X.columns:
            if 'is_credit' in X.columns:
                X['amt_credit_interaction'] = X['TransactionAmt'] * X['is_credit']
            if 'is_visa' in X.columns:
                X['amt_visa_interaction'] = X['TransactionAmt'] * X['is_visa']

        # Time × Card interactions
        if 'hour' in X.columns and 'is_credit' in X.columns:
            X['hour_credit_interaction'] = X['hour'] * X['is_credit']

        # Device × Time interactions
        if 'is_mobile' in X.columns and 'hour' in X.columns:
            X['mobile_hour_interaction'] = X['is_mobile'] * X['hour']

        # Card × Email interactions
        if 'is_credit' in X.columns and 'P_email_is_common' in X.columns:
            X['credit_common_email'] = X['is_credit'] * X['P_email_is_common']

        return X

    def _add_anomaly_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add anomaly detection features."""
        # Transaction amount outliers
        if 'TransactionAmt' in X.columns:
            q1 = X['TransactionAmt'].quantile(0.25)
            q3 = X['TransactionAmt'].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            X['is_amount_outlier'] = ((X['TransactionAmt'] < lower_bound) |
                                       (X['TransactionAmt'] > upper_bound)).astype(int)

        # Time-based anomalies (requires sorting by TransactionDT per entity)
        # This is a simplified version
        if 'TransactionDT' in X.columns:
            # Very rapid transactions (less than 10 seconds)
            time_diff = X['TransactionDT'].diff()
            X['is_rapid_transaction'] = (time_diff < 10).astype(int)

        # Geographic anomalies
        if 'dist1' in X.columns:
            dist_mean = X['dist1'].mean()
            dist_std = X['dist1'].std()
            X['is_dist_outlier'] = (abs(X['dist1'] - dist_mean) > 3 * dist_std).astype(int)

        return X

    def _add_missing_indicators(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add missing value indicators for all columns."""
        for col in X.columns:
            if X[col].isna().any():
                # Use shorter indicator names
                indicator_name = f"{col}_na"
                X[indicator_name] = X[col].isna().astype(int)

        # Also add total missing count
        X['total_missing_count'] = X.isna().sum(axis=1)
        X['total_missing_ratio'] = X['total_missing_count'] / X.shape[1]

        return X

    def get_feature_summary(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Get summary of engineered features."""
        summary = {
            'original_features': X.shape[1],
            'engineered_features': None,  # Will be filled after transform
            'feature_groups_applied': list(self.feature_config.keys()),
        }
        return summary


# Example usage
if __name__ == "__main__":
    # Load your data
    # X, y = load_fraud_data(...)

    # Create feature engineer with custom configuration
    feature_engineer = EnhancedFeatureEngineer(
        feature_config={
            'time_features': True,
            'transaction_features': True,
            'card_features': True,
            'geo_features': False,  # Turn off geographic features
            'email_features': True,
            'device_features': True,
            'id_features': True,
            'v_features': True,
            'interaction_features': True,
            'anomaly_features': False,
            'missing_indicators': True,
        }
    )

    # Fit and transform
    # X_engineered = feature_engineer.fit_transform(X, y)

    print("EnhancedFeatureEngineer class ready for use.")
    print("Available feature groups:")
    for group in feature_engineer.feature_config.keys():
        print(f"  - {group}")