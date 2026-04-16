from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from preprocessor import FraudPreprocessor
from enhanced_feature_engineer import EnhancedFeatureEngineer
from typing import Dict, Optional


def create_enhanced_log_reg_pipeline(feature_config: Optional[Dict[str, bool]] = None) -> Pipeline:
    """
    Create enhanced Logistic Regression pipeline with feature engineering.

    Args:
        feature_config: Configuration for EnhancedFeatureEngineer

    Returns:
        Pipeline with feature engineering, preprocessing, and logistic regression
    """
    pipeline = Pipeline([
        ("features", EnhancedFeatureEngineer(feature_config=feature_config)),
        ("preprocess", FraudPreprocessor()),
        ("model", LogisticRegression(max_iter=1000, random_state=42))
    ])
    return pipeline


def create_enhanced_rf_pipeline(feature_config: Optional[Dict[str, bool]] = None) -> Pipeline:
    """
    Create enhanced Random Forest pipeline with feature engineering.

    Args:
        feature_config: Configuration for EnhancedFeatureEngineer

    Returns:
        Pipeline with feature engineering, preprocessing, and random forest
    """
    pipeline = Pipeline([
        ("features", EnhancedFeatureEngineer(feature_config=feature_config)),
        ("preprocess", FraudPreprocessor()),
        ("model", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight='balanced'  # Important for fraud detection
        ))
    ])
    return pipeline


def create_enhanced_gb_pipeline(feature_config: Optional[Dict[str, bool]] = None) -> Pipeline:
    """
    Create enhanced Gradient Boosting pipeline with feature engineering.

    Args:
        feature_config: Configuration for EnhancedFeatureEngineer

    Returns:
        Pipeline with feature engineering, preprocessing, and gradient boosting
    """
    pipeline = Pipeline([
        ("features", EnhancedFeatureEngineer(feature_config=feature_config)),
        ("preprocess", FraudPreprocessor()),
        ("model", GradientBoostingClassifier(
            n_estimators=200,
            random_state=42
        ))
    ])
    return pipeline


def create_all_enhanced_pipelines(feature_config: Optional[Dict[str, bool]] = None) -> Dict[str, Pipeline]:
    """
    Create all three enhanced pipelines for ensemble training.

    Args:
        feature_config: Configuration for EnhancedFeatureEngineer

    Returns:
        Dictionary with all three pipelines
    """
    pipelines = {
        "LogReg": create_enhanced_log_reg_pipeline(feature_config),
        "RandomForest": create_enhanced_rf_pipeline(feature_config),
        "GradientBoosting": create_enhanced_gb_pipeline(feature_config)
    }
    return pipelines


# Pre-configured feature configurations for different use cases
BASIC_FEATURE_CONFIG = {
    'time_features': True,
    'transaction_features': True,
    'card_features': True,
    'geo_features': True,
    'email_features': True,
    'device_features': False,
    'id_features': False,
    'v_features': True,
    'interaction_features': True,
    'anomaly_features': False,
    'missing_indicators': True,
}

FULL_FEATURE_CONFIG = {
    'time_features': True,
    'transaction_features': True,
    'card_features': True,
    'geo_features': True,
    'email_features': True,
    'device_features': True,
    'id_features': True,
    'v_features': True,
    'interaction_features': True,
    'anomaly_features': True,
    'missing_indicators': True,
}

MINIMAL_FEATURE_CONFIG = {
    'time_features': True,
    'transaction_features': True,
    'card_features': True,
    'geo_features': False,
    'email_features': False,
    'device_features': False,
    'id_features': False,
    'v_features': False,
    'interaction_features': False,
    'anomaly_features': False,
    'missing_indicators': True,
}

# Pre-created pipelines with basic configuration (for backward compatibility)
log_reg_pipeline_enhanced = create_enhanced_log_reg_pipeline(BASIC_FEATURE_CONFIG)
rf_pipeline_enhanced = create_enhanced_rf_pipeline(BASIC_FEATURE_CONFIG)
gb_pipeline_enhanced = create_enhanced_gb_pipeline(BASIC_FEATURE_CONFIG)

all_pipelines_enhanced = create_all_enhanced_pipelines(BASIC_FEATURE_CONFIG)


def compare_pipeline_features(pipeline1: Pipeline, pipeline2: Pipeline, X_sample) -> Dict:
    """
    Compare feature counts between two pipelines.

    Args:
        pipeline1: First pipeline
        pipeline2: Second pipeline
        X_sample: Sample data to transform

    Returns:
        Dictionary with comparison metrics
    """
    # Transform sample data with both pipelines
    X1 = pipeline1.named_steps['features'].fit_transform(X_sample) if 'features' in pipeline1.named_steps else X_sample
    X2 = pipeline2.named_steps['features'].fit_transform(X_sample) if 'features' in pipeline2.named_steps else X_sample

    comparison = {
        'pipeline1_features': X1.shape[1],
        'pipeline2_features': X2.shape[1],
        'difference': X2.shape[1] - X1.shape[1],
        'pipeline1_feature_engineer': pipeline1.named_steps.get('features', None),
        'pipeline2_feature_engineer': pipeline2.named_steps.get('features', None)
    }

    return comparison


# Example usage
if __name__ == "__main__":
    print("Enhanced Pipeline Module")
    print("=" * 60)

    # Show available pipelines
    print("\nAvailable enhanced pipelines:")
    for name, pipeline in all_pipelines_enhanced.items():
        steps = list(pipeline.named_steps.keys())
        print(f"  {name}: {steps}")

    # Show feature configurations
    print("\nPre-configured feature sets:")
    print(f"  BASIC: {len([k for k, v in BASIC_FEATURE_CONFIG.items() if v])} feature groups enabled")
    print(f"  FULL: {len([k for k, v in FULL_FEATURE_CONFIG.items() if v])} feature groups enabled")
    print(f"  MINIMAL: {len([k for k, v in MINIMAL_FEATURE_CONFIG.items() if v])} feature groups enabled")

    print("\nUsage example:")
    print("""
# Import and use enhanced pipelines
from enhanced_pipeline import all_pipelines_enhanced

# Train models with enhanced features
for name, pipeline in all_pipelines_enhanced.items():
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

# Or create custom pipelines
from enhanced_pipeline import create_enhanced_log_reg_pipeline

custom_config = {
    'time_features': True,
    'transaction_features': True,
    'card_features': True,
    'geo_features': False,
    # ... other settings
}

custom_pipeline = create_enhanced_log_reg_pipeline(custom_config)
custom_pipeline.fit(X_train, y_train)
    """)