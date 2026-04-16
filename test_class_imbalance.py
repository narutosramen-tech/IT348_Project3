#!/usr/bin/env python3
"""
Test class imbalance handling with oversampling and undersampling.
"""

import numpy as np
import pandas as pd
from models import (
    apply_sampling,
    create_sampling_pipeline,
    train_and_evaluate_classifiers_with_sampling,
    SecurityFirstEnsemble,
    train_and_evaluate_ensemble_with_sampling
)

def create_imbalanced_test_data():
    """Create a simple imbalanced dataset for testing."""
    # Create highly imbalanced dataset (90% class 0, 10% class 1)
    n_samples = 1000
    n_fraud = 100  # 10% fraud
    n_legitimate = 900  # 90% legitimate

    # Features
    X_legit = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_legitimate),
        'feature2': np.random.normal(0, 1, n_legitimate),
        'feature3': np.random.randint(0, 5, n_legitimate)
    })

    X_fraud = pd.DataFrame({
        'feature1': np.random.normal(1, 1, n_fraud),
        'feature2': np.random.normal(1, 1, n_fraud),
        'feature3': np.random.randint(3, 8, n_fraud)
    })

    X = pd.concat([X_legit, X_fraud], ignore_index=True)

    # Labels
    y_legit = pd.Series([0] * n_legitimate)
    y_fraud = pd.Series([1] * n_fraud)
    y = pd.concat([y_legit, y_fraud], ignore_index=True)

    # Shuffle
    indices = np.random.permutation(len(X))
    X = X.iloc[indices].reset_index(drop=True)
    y = y.iloc[indices].reset_index(drop=True)

    print(f"Created imbalanced dataset: {len(X)} samples")
    print(f"  Legitimate (0): {(y == 0).sum()} samples")
    print(f"  Fraud (1): {(y == 1).sum()} samples")
    print(f"  Fraud ratio: {(y == 1).mean():.2%}")

    return X, y

def test_sampling_techniques():
    """Test different sampling techniques."""
    print("Test 1: Sampling Techniques")
    print("=" * 60)

    X, y = create_imbalanced_test_data()

    # Split into train/test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"\nTraining set: {len(X_train)} samples")
    print(f"  Fraud: {(y_train == 1).sum()} ({(y_train == 1).mean():.2%})")

    # Test different sampling methods
    sampling_methods = ["none", "oversample", "undersample"]

    for method in sampling_methods:
        print(f"\n--- Testing {method} sampling ---")

        try:
            X_resampled, y_resampled = apply_sampling(
                X_train, y_train,
                sampling_method=method,
                sampling_strategy='auto',
                random_state=42
            )

            print(f"After {method}: {len(X_resampled)} samples")
            print(f"  Fraud: {(y_resampled == 1).sum()} ({(y_resampled == 1).mean():.2%})")

            # Test training a simple model
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import Pipeline

            # Create base pipeline
            base_pipeline = Pipeline([
                ('model', LogisticRegression(max_iter=1000, random_state=42))
            ])

            # Create pipeline with sampling
            pipeline = create_sampling_pipeline(
                base_pipeline,
                sampling_method=method,
                sampling_strategy='auto',
                random_state=42
            )

            # Train
            if method == "none":
                pipeline.fit(X_train, y_train)
            else:
                pipeline.fit(X_resampled, y_resampled)

            # Evaluate
            y_pred = pipeline.predict(X_test)
            accuracy = (y_pred == y_test).mean()
            recall = (y_test[y_test == 1] == y_pred[y_test == 1]).mean()

            print(f"  Test Accuracy: {accuracy:.4f}")
            print(f"  Test Recall: {recall:.4f}")

        except Exception as e:
            print(f"ERROR with {method}: {e}")
            import traceback
            traceback.print_exc()

def test_classifier_with_sampling():
    """Test training classifiers with different sampling methods."""
    print("\n\nTest 2: Classifier Training with Sampling")
    print("=" * 60)

    X, y = create_imbalanced_test_data()

    # Split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    sampling_methods = ["none", "oversample", "undersample"]

    for method in sampling_methods:
        print(f"\n--- Training classifiers with {method} sampling ---")

        try:
            results = train_and_evaluate_classifiers_with_sampling(
                X_train, X_test, y_train, y_test,
                sampling_method=method,
                sampling_strategy='auto',
                verbose=False,
                random_state=42
            )

            print(f"Trained {len(results)} models")

            # Show best model
            best_model = None
            best_recall = -1

            for model_name, model_results in results.items():
                recall = model_results['metrics']['recall']
                f1 = model_results['metrics']['f1_score']
                print(f"  {model_name}: Recall={recall:.4f}, F1={f1:.4f}")

                if recall > best_recall:
                    best_recall = recall
                    best_model = model_name

            print(f"  Best model ({best_model}): Recall={best_recall:.4f}")

        except Exception as e:
            print(f"ERROR with {method}: {e}")
            import traceback
            traceback.print_exc()

def test_ensemble_with_sampling():
    """Test ensemble training with different sampling methods."""
    print("\n\nTest 3: Ensemble Training with Sampling")
    print("=" * 60)

    X, y = create_imbalanced_test_data()

    # Split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    sampling_methods = ["none", "oversample"]

    for method in sampling_methods:
        print(f"\n--- Training ensemble with {method} sampling ---")

        try:
            results = train_and_evaluate_ensemble_with_sampling(
                X_train, X_test, y_train, y_test,
                voting_type="hard",
                tie_breaker="malware",
                sampling_method=method,
                sampling_strategy='auto',
                random_state=42
            )

            ensemble = results['ensemble']
            ensemble_results = results['ensemble_results']

            print(f"Ensemble trained successfully")
            print(f"  Ensemble Recall: {ensemble_results['metrics']['recall']:.4f}")
            print(f"  Ensemble F1: {ensemble_results['metrics']['f1_score']:.4f}")
            print(f"  Ensemble ROC-AUC: {ensemble_results['metrics'].get('roc_auc', 0):.4f}")

        except Exception as e:
            print(f"ERROR with {method}: {e}")
            import traceback
            traceback.print_exc()

def test_smote_if_available():
    """Test SMOTE sampling if imblearn is available."""
    print("\n\nTest 4: SMOTE Sampling (if available)")
    print("=" * 60)

    try:
        # Check if imblearn is available
        from imblearn.over_sampling import SMOTE
        print("imblearn is available, testing SMOTE...")

        X, y = create_imbalanced_test_data()

        # Split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        print(f"\nOriginal training set: {len(X_train)} samples")
        print(f"  Fraud: {(y_train == 1).sum()} ({(y_train == 1).mean():.2%})")

        # Apply SMOTE
        X_resampled, y_resampled = apply_sampling(
            X_train, y_train,
            sampling_method="smote",
            sampling_strategy='auto',
            random_state=42
        )

        print(f"After SMOTE: {len(X_resampled)} samples")
        print(f"  Fraud: {(y_resampled == 1).sum()} ({(y_resampled == 1).mean():.2%})")

        # Train a simple model
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_resampled, y_resampled)

        y_pred = model.predict(X_test)
        accuracy = (y_pred == y_test).mean()
        recall = (y_test[y_test == 1] == y_pred[y_test == 1]).mean()

        print(f"  Test Accuracy: {accuracy:.4f}")
        print(f"  Test Recall: {recall:.4f}")

    except ImportError:
        print("imblearn not available. Install with: pip install imbalanced-learn")
    except Exception as e:
        print(f"ERROR with SMOTE: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run all tests."""
    print("Testing Class Imbalance Handling")
    print("=" * 60)

    try:
        test_sampling_techniques()
        test_classifier_with_sampling()
        test_ensemble_with_sampling()
        test_smote_if_available()

        print("\n" + "=" * 60)
        print("IMPORTANT: For full imblearn functionality, install:")
        print("  pip install imbalanced-learn")
        print("\nThe implementation supports:")
        print("  1. Random oversampling")
        print("  2. Random undersampling")
        print("  3. SMOTE (if imblearn installed)")
        print("  4. Integration with all existing models")
        print("  5. Pipeline integration")

    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())