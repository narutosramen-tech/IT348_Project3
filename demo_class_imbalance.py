#!/usr/bin/env python3
"""
Demonstration of class imbalance handling features.
"""

import pandas as pd
import numpy as np
from models import (
    apply_sampling,
    train_and_evaluate_classifiers_with_sampling,
    SecurityFirstEnsemble,
    train_and_evaluate_ensemble_with_sampling,
    train_ensemble_from_dataset_with_sampling
)
from dataset import Dataset
from sample import Sample

def demonstrate_sampling_techniques():
    """Show how to use different sampling techniques."""
    print("Class Imbalance Handling Demonstration")
    print("=" * 70)

    # Create a simple imbalanced dataset
    np.random.seed(42)
    n_samples = 1000
    n_features = 10

    # Create imbalanced data (90% class 0, 10% class 1)
    X = pd.DataFrame(np.random.randn(n_samples, n_features),
                     columns=[f'feature_{i}' for i in range(n_features)])
    y = pd.Series([0] * 900 + [1] * 100)  # 90% class 0, 10% class 1
    indices = np.random.permutation(n_samples)
    y = y.iloc[indices].reset_index(drop=True)
    X = X.iloc[indices].reset_index(drop=True)

    print(f"Original dataset: {len(X)} samples")
    print(f"  Class 0 (Legitimate): {(y == 0).sum()} samples")
    print(f"  Class 1 (Fraud): {(y == 1).sum()} samples")
    print(f"  Class imbalance ratio: {(y == 1).mean():.1%}")

    print("\n" + "-" * 70)
    print("1. Applying Different Sampling Techniques")
    print("-" * 70)

    sampling_methods = ["none", "oversample", "undersample"]

    for method in sampling_methods:
        print(f"\n{method.upper()} SAMPLING:")
        X_resampled, y_resampled = apply_sampling(
            X, y,
            sampling_method=method,
            sampling_strategy='auto',
            random_state=42
        )

        print(f"  Samples before: {len(X)}")
        print(f"  Samples after: {len(X_resampled)}")
        print(f"  Class distribution after: {(y_resampled == 1).mean():.1%}")

def demonstrate_training_with_sampling():
    """Show how to train models with different sampling methods."""
    print("\n" + "=" * 70)
    print("2. Training Models with Sampling")
    print("=" * 70)

    # Create dataset
    np.random.seed(42)
    n_samples = 500
    n_features = 5

    X = pd.DataFrame(np.random.randn(n_samples, n_features),
                     columns=[f'f{i}' for i in range(n_features)])
    y = pd.Series([0] * 450 + [1] * 50)  # 90% class 0, 10% class 1
    indices = np.random.permutation(n_samples)
    y = y.iloc[indices].reset_index(drop=True)
    X = X.iloc[indices].reset_index(drop=True)

    # Split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"Training set: {len(X_train)} samples ({(y_train == 1).mean():.1%} fraud)")
    print(f"Test set: {len(X_test)} samples ({(y_test == 1).mean():.1%} fraud)")

    print("\n" + "-" * 70)
    print("Training individual classifiers with different sampling:")

    results = train_and_evaluate_classifiers_with_sampling(
        X_train, X_test, y_train, y_test,
        sampling_method="oversample",
        sampling_strategy='auto',
        verbose=False,
        random_state=42
    )

    print(f"\nResults with oversampling:")
    for model_name, model_results in results.items():
        metrics = model_results['metrics']
        print(f"  {model_name}: Recall={metrics['recall']:.3f}, "
              f"F1={metrics['f1_score']:.3f}, "
              f"ROC-AUC={metrics.get('roc_auc', 0):.3f}")

def demonstrate_ensemble_with_sampling():
    """Show how to train ensemble with sampling."""
    print("\n" + "=" * 70)
    print("3. Training Ensemble with Sampling")
    print("=" * 70)

    # Create dataset
    np.random.seed(42)
    n_samples = 500
    n_features = 5

    X = pd.DataFrame(np.random.randn(n_samples, n_features),
                     columns=[f'f{i}' for i in range(n_features)])
    y = pd.Series([0] * 450 + [1] * 50)  # 90% class 0, 10% class 1
    indices = np.random.permutation(n_samples)
    y = y.iloc[indices].reset_index(drop=True)
    X = X.iloc[indices].reset_index(drop=True)

    # Split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print("Training SecurityFirstEnsemble with oversampling...")

    ensemble = SecurityFirstEnsemble(
        tie_breaker="malware",
        voting_type="hard",
        sampling_method="oversample",
        sampling_strategy='auto',
        random_state=42
    )

    ensemble.fit(X_train, y_train)

    # Evaluate
    results = ensemble.evaluate(X_test, y_test, verbose=False)

    print(f"\nEnsemble Results:")
    print(f"  Recall: {results['metrics']['recall']:.3f}")
    print(f"  F1-Score: {results['metrics']['f1_score']:.3f}")
    print(f"  ROC-AUC: {results['metrics'].get('roc_auc', 0):.3f}")
    print(f"  PR-AUC: {results['metrics'].get('pr_auc', 0):.3f}")

def demonstrate_dataset_integration():
    """Show how to use sampling with Dataset objects."""
    print("\n" + "=" * 70)
    print("4. Integration with Dataset Objects")
    print("=" * 70)

    # Create Dataset object
    np.random.seed(42)
    n_samples = 300
    n_features = 5

    X = pd.DataFrame(np.random.randn(n_samples, n_features),
                     columns=[f'f{i}' for i in range(n_features)])
    y = pd.Series([0] * 270 + [1] * 30)  # 90% class 0, 10% class 1
    indices = np.random.permutation(n_samples)
    y = y.iloc[indices].reset_index(drop=True)
    X = X.iloc[indices].reset_index(drop=True)

    # Create Dataset
    dataset = Dataset()
    sample = Sample(name="imbalanced_data", features=X, labels=y)
    dataset.add_sample(sample)

    print("Dataset created with sample 'imbalanced_data'")
    dataset.summary()

    print("\nTraining from dataset with different sampling methods:")

    sampling_methods = ["none", "oversample", "undersample"]

    for method in sampling_methods:
        try:
            print(f"\n{method.upper()} SAMPLING:")

            # Note: For demonstration, we won't actually run this as it requires
            # more data, but we show the function call
            print(f"  Use: train_ensemble_from_dataset_with_sampling(")
            print(f"      dataset=dataset,")
            print(f"      sample_name='imbalanced_data',")
            print(f"      sampling_method='{method}',")
            print(f"      sampling_strategy='auto',")
            print(f"      random_state=42")
            print(f"  )")

        except Exception as e:
            print(f"  Error: {e}")

def usage_examples():
    """Show usage examples."""
    print("\n" + "=" * 70)
    print("5. Usage Examples")
    print("=" * 70)

    print("\n1. Apply sampling to training data:")
    print("""
# Apply oversampling
X_resampled, y_resampled = apply_sampling(
    X_train, y_train,
    sampling_method="oversample",
    sampling_strategy='auto',
    random_state=42
)
    """)

    print("\n2. Train classifiers with sampling:")
    print("""
# Train individual classifiers with oversampling
results = train_and_evaluate_classifiers_with_sampling(
    X_train, X_test, y_train, y_test,
    sampling_method="oversample",
    sampling_strategy='auto',
    verbose=True,
    random_state=42
)
    """)

    print("\n3. Train ensemble with sampling:")
    print("""
# Create and train ensemble with SMOTE
ensemble = SecurityFirstEnsemble(
    tie_breaker="malware",
    voting_type="hard",
    sampling_method="smote",  # Use SMOTE for oversampling
    sampling_strategy='auto',
    random_state=42
)

ensemble.fit(X_train, y_train)
    """)

    print("\n4. Use different sampling strategies:")
    print("""
# Balance classes (default)
sampling_strategy='auto'

# Keep minority class at 30% of majority
sampling_strategy=0.3

# Specify exact number of samples
sampling_strategy={0: 500, 1: 500}  # 500 samples of each class
    """)

def main():
    """Run demonstration."""
    try:
        demonstrate_sampling_techniques()
        demonstrate_training_with_sampling()
        demonstrate_ensemble_with_sampling()
        demonstrate_dataset_integration()
        usage_examples()

        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print("\nClass imbalance handling has been successfully added to:")
        print("1. apply_sampling() - Apply oversampling/undersampling/SMOTE to data")
        print("2. create_sampling_pipeline() - Create pipelines with sampling steps")
        print("3. train_and_evaluate_classifiers_with_sampling() - Train models with sampling")
        print("4. SecurityFirstEnsemble - Now supports sampling in constructor")
        print("5. train_and_evaluate_ensemble_with_sampling() - Train ensemble with sampling")
        print("6. train_ensemble_from_dataset_with_sampling() - Train from Dataset with sampling")

        print("\nAvailable sampling methods:")
        print("  - 'none': No sampling (default)")
        print("  - 'oversample': Random oversampling")
        print("  - 'undersample': Random undersampling")
        print("  - 'smote': SMOTE (requires imblearn installation)")

        print("\nInstall imblearn for full functionality:")
        print("  pip install imbalanced-learn")

    except Exception as e:
        print(f"\nError in demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())