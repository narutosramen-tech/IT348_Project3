# Feature Engineering Suggestions for Fraud Detection

Based on the current `FeatureEngineer` class and the data structure, here are comprehensive suggestions for feature engineering.

## Current Implementation Analysis

The current `FeatureEngineer` class does:
1. **Time features**: Extracts hour and day from `TransactionDT`
2. **Log transform**: Applies log1p to `TransactionAmt`
3. **Missing indicators**: Creates binary indicators for missing values

## Suggested Feature Engineering Additions

### 1. **Time-Based Features**
```python
# Already implemented:
# X['hour'] = (X['TransactionDT'] // 3600) % 24
# X['day'] = (X['TransactionDT'] // 86400)

# Additional suggestions:
X['weekday'] = (X['day'] % 7)  # Day of week (0-6)
X['is_weekend'] = X['weekday'].isin([5, 6]).astype(int)  # Weekend indicator
X['hour_sin'] = np.sin(2 * np.pi * X['hour'] / 24)  # Cyclical encoding
X['hour_cos'] = np.cos(2 * np.pi * X['hour'] / 24)
X['hour_bin'] = pd.cut(X['hour'], bins=[0, 6, 12, 18, 24], labels=['night', 'morning', 'afternoon', 'evening'])
X['is_night'] = ((X['hour'] >= 0) & (X['hour'] <= 6)).astype(int)  # High risk hours
X['hour_of_day'] = X['hour']  # Already have, but keep as reference
X['transaction_time_seconds'] = X['TransactionDT'] % 3600  # Seconds within the hour
```

### 2. **Transaction Amount Features**
```python
# Current: X['TransactionAmt_log'] = np.log1p(X['TransactionAmt'])

# Additional:
X['TransactionAmt_binned'] = pd.qcut(X['TransactionAmt'], q=10, labels=False)  # Quantile bins
X['TransactionAmt_zscore'] = (X['TransactionAmt'] - X['TransactionAmt'].mean()) / X['TransactionAmt'].std()
X['TransactionAmt_percentile'] = X['TransactionAmt'].rank(pct=True)
X['is_high_amount'] = (X['TransactionAmt'] > X['TransactionAmt'].quantile(0.95)).astype(int)
X['TransactionAmt_sqrt'] = np.sqrt(X['TransactionAmt'])
X['TransactionAmt_boxcox'] = boxcox(X['TransactionAmt'] + 1)[0]  # Requires scipy
```

### 3. **Card-Based Features**
```python
# Card features from transaction data
if 'card1' in X:
    X['card_digits'] = X['card1'].astype(str).str[:4]  # First 4 digits (BIN)
    X['card_bin'] = X['card1'].astype(str).str[:6]  # BIN code
    
if 'card2' in X and 'card3' in X:
    X['card2_card3_diff'] = X['card2'] - X['card3']
    X['card_combination'] = X['card2'].astype(str) + '_' + X['card3'].astype(str)
    
if 'card4' in X:
    # One-hot encode card type (Visa, MasterCard, etc.)
    X['is_visa'] = (X['card4'] == 'visa').astype(int)
    X['is_mastercard'] = (X['card4'] == 'mastercard').astype(int)
    # etc...

if 'card6' in X:
    X['is_debit'] = (X['card6'] == 'debit').astype(int)
    X['is_credit'] = (X['card6'] == 'credit').astype(int)
```

### 4. **Geographic Features**
```python
if 'addr1' in X and 'addr2' in X:
    X['addr1_addr2_combo'] = X['addr1'].astype(str) + '_' + X['addr2'].astype(str)
    X['addr1_is_missing'] = X['addr1'].isna().astype(int)
    X['addr2_is_missing'] = X['addr2'].isna().astype(int)
    X['both_addr_missing'] = (X['addr1_is_missing'] & X['addr2_is_missing']).astype(int)
    
if 'dist1' in X and 'dist2' in X:
    X['dist_abs_diff'] = abs(X['dist1'] - X['dist2'])
    X['dist_sum'] = X['dist1'] + X['dist2']
    X['dist_ratio'] = X['dist1'] / (X['dist2'] + 1e-6)  # Avoid division by zero
    X['is_same_location'] = (X['dist1'] == X['dist2']).astype(int)
```

### 5. **Email Domain Features**
```python
if 'P_emaildomain' in X:
    # Common email providers
    common_domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'aol.com']
    X['P_email_is_common'] = X['P_emaildomain'].isin(common_domains).astype(int)
    X['P_email_is_free'] = X['P_emaildomain'].str.contains('gmail|yahoo|hotmail|outlook|aol', case=False).astype(int)
    X['P_email_domain_length'] = X['P_emaildomain'].str.len()
    # Extract TLD
    X['P_email_tld'] = X['P_emaildomain'].str.extract(r'\.([^.]+)$')
    
if 'R_emaildomain' in X:
    X['R_email_is_common'] = X['R_emaildomain'].isin(common_domains).astype(int)
    X['email_match'] = (X['P_emaildomain'] == X['R_emaildomain']).astype(int)
    X['email_both_missing'] = (X['P_emaildomain'].isna() & X['R_emaildomain'].isna()).astype(int)
```

### 6. **Device and Identity Features** (from identity.csv after merge)
```python
# After merging with identity data
if 'DeviceType' in X:
    X['is_mobile'] = (X['DeviceType'] == 'mobile').astype(int)
    X['is_desktop'] = (X['DeviceType'] == 'desktop').astype(int)
    
if 'DeviceInfo' in X:
    # Extract device brand/model
    X['device_is_android'] = X['DeviceInfo'].str.contains('Android', case=False).astype(int)
    X['device_is_ios'] = X['DeviceInfo'].str.contains('iOS|iPhone', case=False).astype(int)
    X['device_is_windows'] = X['DeviceInfo'].str.contains('Windows', case=False).astype(int)
    X['device_is_mac'] = X['DeviceInfo'].str.contains('Mac|macOS', case=False).astype(int)
    
# ID feature interactions
id_cols = [col for col in X.columns if col.startswith('id_')]
if id_cols:
    X['id_missing_count'] = X[id_cols].isna().sum(axis=1)
    X['id_filled_count'] = len(id_cols) - X['id_missing_count']
    X['id_filled_ratio'] = X['id_filled_count'] / len(id_cols)
    
    # For numeric ID columns
    numeric_id_cols = [col for col in id_cols if X[col].dtype in ['float64', 'int64']]
    if numeric_id_cols:
        X['id_numeric_mean'] = X[numeric_id_cols].mean(axis=1, skipna=True)
        X['id_numeric_std'] = X[numeric_id_cols].std(axis=1, skipna=True)
```

### 7. **Aggregate Features** (requires feature store or rolling windows)
```python
# These require maintaining transaction history per entity
# Could be implemented with a FeatureStore class

# Per card features
# card_transaction_count_last_24h
# card_avg_amount_last_7d
# card_fraud_rate_last_30d

# Per email features  
# email_transaction_count_last_24h
# email_unique_cards_last_7d

# Per device features
# device_transaction_count_last_1h
# device_unique_users_last_24h
```

### 8. **Interaction Features**
```python
# Cross-feature interactions
if 'TransactionAmt' in X and 'hour' in X:
    X['amount_hour_interaction'] = X['TransactionAmt'] * X['hour']
    
if 'TransactionAmt' in X and 'is_weekend' in X:
    X['weekend_high_amount'] = (X['is_weekend'] & X['is_high_amount']).astype(int)
    
if 'card1' in X and 'addr1' in X:
    X['card_addr_combo'] = X['card1'].astype(str) + '_' + X['addr1'].astype(str)
    
if 'ProductCD' in X and 'card4' in X:
    X['product_card_combo'] = X['ProductCD'] + '_' + X['card4'].astype(str)
```

### 9. **Statistical Features from V Columns**
```python
# V columns appear to be V1-V339 (likely PCA or other engineered features)
v_cols = [col for col in X.columns if col.startswith('V')]
if v_cols:
    # Basic statistics
    X['V_mean'] = X[v_cols].mean(axis=1, skipna=True)
    X['V_std'] = X[v_cols].std(axis=1, skipna=True)
    X['V_sum'] = X[v_cols].sum(axis=1, skipna=True)
    X['V_min'] = X[v_cols].min(axis=1, skipna=True)
    X['V_max'] = X[v_cols].max(axis=1, skipna=True)
    X['V_skew'] = X[v_cols].skew(axis=1, skipna=True)
    
    # Missing values in V columns
    X['V_missing_count'] = X[v_cols].isna().sum(axis=1)
    X['V_missing_ratio'] = X['V_missing_count'] / len(v_cols)
    
    # Non-zero counts
    X['V_non_zero_count'] = (X[v_cols] != 0).sum(axis=1)
```

### 10. **Categorical Encoding Improvements**
```python
# For categorical columns like ProductCD, card4, card6, etc.
cat_cols = ['ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain', 'DeviceType']
for col in cat_cols:
    if col in X:
        # Frequency encoding
        freq = X[col].value_counts(normalize=True)
        X[f'{col}_freq'] = X[col].map(freq)
        
        # Target encoding (requires y for training)
        # Would need to be implemented separately for train/test
```

### 11. **Anomaly Detection Features**
```python
# Statistical outliers
if 'TransactionAmt' in X:
    q1 = X['TransactionAmt'].quantile(0.25)
    q3 = X['TransactionAmt'].quantile(0.75)
    iqr = q3 - q1
    X['is_amount_outlier'] = ((X['TransactionAmt'] < (q1 - 1.5 * iqr)) | 
                              (X['TransactionAmt'] > (q3 + 1.5 * iqr))).astype(int)

# Time anomaly
if 'TransactionDT' in X:
    X['time_since_last_txn'] = X['TransactionDT'].diff()  # Requires sorting by TransactionDT per entity
    X['time_anomaly'] = (X['time_since_last_txn'] < 10).astype(int)  # Very rapid transactions
```

### 12. **Network Graph Features** (advanced)
```python
# These would require building a transaction graph
# card1 -> addr1 connections
# email -> card connections  
# device -> card connections

# Features could include:
# - Degree centrality of nodes
# - Clustering coefficient
# - Number of unique connections
# - Transaction patterns in subgraphs
```

## Implementation Strategy

### Option 1: Enhanced FeatureEngineer Class
```python
class EnhancedFeatureEngineer:
    def __init__(self, feature_config: dict = None):
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
            'anomaly_features': True
        }
        self.fitted = False
        # Store statistics for transformations
        self.amt_mean = None
        self.amt_std = None
        self.email_domains = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # Compute statistics needed for transformations
        if 'TransactionAmt' in X:
            self.amt_mean = X['TransactionAmt'].mean()
            self.amt_std = X['TransactionAmt'].std()
        
        if 'P_emaildomain' in X:
            common_domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'aol.com']
            self.email_domains = common_domains
            
        self.fitted = True
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        
        # Time features
        if self.feature_config.get('time_features') and 'TransactionDT' in X:
            self._add_time_features(X)
            
        # Transaction amount features
        if self.feature_config.get('transaction_features') and 'TransactionAmt' in X:
            self._add_transaction_features(X)
            
        # Card features
        if self.feature_config.get('card_features'):
            self._add_card_features(X)
            
        # ... etc for each feature category
            
        return X
    
    def _add_time_features(self, X):
        X['hour'] = (X['TransactionDT'] // 3600) % 24
        X['day'] = (X['TransactionDT'] // 86400)
        X['weekday'] = (X['day'] % 7)
        X['is_weekend'] = X['weekday'].isin([5, 6]).astype(int)
        X['hour_sin'] = np.sin(2 * np.pi * X['hour'] / 24)
        X['hour_cos'] = np.cos(2 * np.pi * X['hour'] / 24)
        X['is_night'] = ((X['hour'] >= 0) & (X['hour'] <= 6)).astype(int)
        
    def _add_transaction_features(self, X):
        X['TransactionAmt_log'] = np.log1p(X['TransactionAmt'])
        X['TransactionAmt_zscore'] = (X['TransactionAmt'] - self.amt_mean) / self.amt_std
        X['is_high_amount'] = (X['TransactionAmt'] > X['TransactionAmt'].quantile(0.95)).astype(int)
        
    # ... other _add_* methods
```

### Option 2: Modular Feature Engineering
Create separate classes for different feature types:
```python
class TimeFeatureEngineer:
    def transform(self, X):
        # Add time-based features
        pass

class CardFeatureEngineer:
    def transform(self, X):
        # Add card-based features
        pass

class CompositeFeatureEngineer:
    def __init__(self):
        self.engineers = [
            TimeFeatureEngineer(),
            CardFeatureEngineer(),
            # ... others
        ]
    
    def transform(self, X):
        for engineer in self.engineers:
            X = engineer.transform(X)
        return X
```

## Next Steps

1. **Start with high-impact features**: Time, amount, and card features usually have highest ROI
2. **Test incrementally**: Add features one category at a time and measure impact on model performance
3. **Monitor feature importance**: Use your `ClassifierEvaluator` and model feature importance
4. **Consider computational cost**: Some features (like graph features) are expensive to compute
5. **Handle data leakage**: Ensure features don't leak future information
6. **Implement feature selection**: After creating many features, select the most important ones

## Integration with Existing Code

Your current pipeline uses `FraudPreprocessor` and likely the `FeatureEngineer`. You could:
1. Enhance the existing `FeatureEngineer` class
2. Create a new `EnhancedFeatureEngineer` class
3. Use the modular approach with different feature engineers
4. Update your `fraud_cli.py` to allow selecting feature engineering strategies

Remember to test each new feature engineering approach with your `SecurityFirstEnsemble` to ensure it improves fraud detection performance, particularly recall (which is most critical for fraud detection).