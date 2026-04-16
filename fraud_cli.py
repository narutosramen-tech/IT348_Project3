#!/usr/bin/env python3
"""
CLI for Fraud Detection Models

A command-line interface to train and test fraud detection models using:
1. SecurityFirstEnsemble from models.py
2. ClassifierEvaluator from models.py
3. Dataset objects from dataset.py
4. Data loading from data.py

Usage Examples:
    # Train with hard voting ensemble
    python fraud_cli.py train --transactions data/transactions.csv --identity data/identity.csv --voting hard

    # Test a trained model
    python fraud_cli.py test --model saved_model.pkl --transactions test_transactions.csv --identity test_identity.csv

    # Compare ensemble vs individual models
    python fraud_cli.py compare --transactions data/transactions.csv --identity data/identity.csv
"""

import argparse
import sys
import pickle
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any

# Import project modules
from data import load_fraud_data
from dataset import Dataset
from sample import Sample
from models import (
    SecurityFirstEnsemble,
    train_and_evaluate_classifiers,
    train_and_evaluate_ensemble,
    ClassifierEvaluator,
    train_ensemble_from_dataset,
    train_from_dataset
)

def load_and_create_dataset(transactions_path: str, identity_path: str, dataset_name: str = "fraud_data") -> Dataset:
    """
    Load fraud data and create a Dataset object with a Sample.

    Args:
        transactions_path: Path to transactions CSV file
        identity_path: Path to identity CSV file
        dataset_name: Name for the dataset sample

    Returns:
        Dataset object containing the loaded data
    """
    print(f"Loading data from:")
    print(f"  Transactions: {transactions_path}")
    print(f"  Identity: {identity_path}")

    X, y = load_fraud_data(transactions_path, identity_path)

    print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Fraud rate: {y.sum()}/{len(y)} ({y.mean():.2%})")

    # Create Dataset and Sample
    dataset = Dataset()
    sample = Sample(name=dataset_name, features=X, labels=y)
    dataset.add_sample(sample)

    return dataset


def train_command(args):
    """Handle the train command."""
    print(f"\n{'='*70}")
    print("TRAINING SECURITY-FIRST ENSEMBLE")
    print(f"{'='*70}")

    # Load data and create dataset
    dataset = load_and_create_dataset(
        args.transactions,
        args.identity,
        dataset_name=args.dataset_name
    )

    # Train ensemble
    print(f"\nTraining ensemble with {args.voting} voting, tie-breaker: {args.tie_breaker}")

    if args.sample_fraction < 1.0:
        print(f"Using {args.sample_fraction:.0%} of data for training...")
        sample = dataset.get(args.dataset_name)
        X, y = sample.get_X_y()

        # Sample the data
        n_samples = int(len(X) * args.sample_fraction)
        indices = np.random.choice(len(X), n_samples, replace=False)
        X_sample = X.iloc[indices].copy()
        y_sample = y.iloc[indices].copy()

        # Create new sample with sampled data
        sampled_dataset = Dataset()
        sampled_sample = Sample(
            name=f"{args.dataset_name}_sampled",
            features=X_sample,
            labels=y_sample
        )
        sampled_dataset.add_sample(sampled_sample)
        dataset = sampled_dataset

    # Train ensemble
    results = train_ensemble_from_dataset(
        dataset=dataset,
        sample_name=args.dataset_name,
        voting_type=args.voting,
        tie_breaker=args.tie_breaker
    )

    ensemble = results['ensemble']
    ensemble_results = results['ensemble_results']

    # Save model if requested
    if args.save_model:
        model_path = Path(args.save_model)
        with open(model_path, 'wb') as f:
            pickle.dump(ensemble, f)
        print(f"\nModel saved to: {model_path}")

    # Save results if requested
    if args.save_results:
        results_path = Path(args.save_results)
        # Convert results to serializable format
        serializable_results = {
            'ensemble_type': ensemble_results.get('ensemble_type'),
            'tie_breaker': ensemble_results.get('tie_breaker'),
            'num_models': ensemble_results.get('num_models'),
            'model_agreement_rate': ensemble_results.get('model_agreement_rate'),
            'metrics': ensemble_results.get('metrics'),
            'improvement': results.get('improvement')
        }

        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"Results saved to: {results_path}")

    return ensemble, results


def test_command(args):
    """Handle the test command."""
    print(f"\n{'='*70}")
    print("TESTING MODEL")
    print(f"{'='*70}")

    # Load trained model
    print(f"Loading model from: {args.model}")
    with open(args.model, 'rb') as f:
        ensemble = pickle.load(f)

    if not isinstance(ensemble, SecurityFirstEnsemble):
        raise TypeError(f"Loaded model is not a SecurityFirstEnsemble: {type(ensemble)}")

    # Load test data
    print(f"Loading test data...")
    X_test, y_test = load_fraud_data(args.transactions, args.identity)
    print(f"Test data: {X_test.shape[0]} samples")

    # Get predictions
    y_pred = ensemble.predict(X_test)

    # Get predicted probabilities if available
    y_pred_proba = None
    try:
        y_pred_proba = ensemble.predict_proba(X_test)[:, 1]
    except Exception as e:
        print(f"Note: Could not get predicted probabilities: {e}")

    # Create evaluator for visualizations
    evaluator = ClassifierEvaluator(
        classifier_name="SecurityFirstEnsemble",
        y_true=y_test,
        y_pred=y_pred,
        y_pred_proba=y_pred_proba
    )

    # Check if we should create save paths
    save_paths = {}
    if args.save_plots_dir:
        import os
        os.makedirs(args.save_plots_dir, exist_ok=True)
        save_paths = {
            'confusion_matrix': os.path.join(args.save_plots_dir, 'ensemble_confusion_matrix.png'),
            'roc_curve': os.path.join(args.save_plots_dir, 'ensemble_roc_curve.png'),
            'pr_curve': os.path.join(args.save_plots_dir, 'ensemble_pr_curve.png'),
            'threshold_analysis': os.path.join(args.save_plots_dir, 'ensemble_threshold_analysis.png')
        }

    # Plot confusion matrix if requested
    if args.plot_confusion_matrix:
        print("\n" + "="*70)
        print("PLOTTING CONFUSION MATRIX")
        print("="*70)
        evaluator.plot_confusion_matrix(
            save_path=save_paths.get('confusion_matrix'),
            normalize=args.normalize_cm
        )

    # Evaluate ensemble
    print(f"\nEvaluating on test data...")
    results = ensemble.evaluate(
        X_test=X_test,
        y_test=y_test,
        verbose=True,
        threshold=args.threshold
    )

    # Additional visualizations
    if args.plot_roc_curve and y_pred_proba is not None:
        print("\n" + "="*70)
        print("PLOTTING ROC CURVE")
        print("="*70)
        evaluator.plot_roc_curve(save_path=save_paths.get('roc_curve'))

    if args.plot_pr_curve and y_pred_proba is not None:
        print("\n" + "="*70)
        print("PLOTTING PRECISION-RECALL CURVE")
        print("="*70)
        evaluator.plot_precision_recall_curve(save_path=save_paths.get('pr_curve'))

    if args.plot_threshold_analysis and y_pred_proba is not None:
        print("\n" + "="*70)
        print("PLOTTING THRESHOLD ANALYSIS")
        print("="*70)
        evaluator.plot_threshold_analysis(save_path=save_paths.get('threshold_analysis'))

    # Generate all visualizations if save directory is specified
    if args.save_plots_dir and y_pred_proba is not None:
        print("\n" + "="*70)
        print("GENERATING ALL VISUALIZATIONS")
        print("="*70)
        evaluator.visualize_all(save_dir=args.save_plots_dir)

    # Save results if requested
    if args.save_results:
        results_path = Path(args.save_results)

        # Convert to serializable format
        serializable_results = {
            'ensemble_type': results.get('ensemble_type'),
            'tie_breaker': results.get('tie_breaker'),
            'num_models': results.get('num_models'),
            'model_agreement_rate': results.get('model_agreement_rate'),
            'metrics': results.get('metrics'),
            'test_samples': X_test.shape[0]
        }

        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"\nTest results saved to: {results_path}")

    return results


def compare_command(args):
    """Handle the compare command."""
    print(f"\n{'='*70}")
    print("COMPARING ENSEMBLE VS INDIVIDUAL MODELS")
    print(f"{'='*70}")

    # Load data
    dataset = load_and_create_dataset(
        args.transactions,
        args.identity,
        dataset_name=args.dataset_name
    )

    print(f"\nTraining individual models...")
    X_train, X_test, y_train, y_test = dataset.train_test_split(args.dataset_name)

    # Train and evaluate individual classifiers
    individual_results = train_and_evaluate_classifiers(
        X_train, X_test, y_train, y_test, verbose=False
    )

    # Train and evaluate ensemble
    print(f"\nTraining ensemble with {args.voting} voting...")
    ensemble = SecurityFirstEnsemble(
        tie_breaker=args.tie_breaker,
        voting_type=args.voting
    )
    ensemble.fit(X_train, y_train)

    # Get ensemble predictions
    y_pred_ensemble = ensemble.predict(X_test)

    # Create evaluators for comparison
    evaluators = {}

    # Individual model evaluators
    for model_name, model_data in individual_results.items():
        y_pred = model_data['model'].predict(X_test)
        evaluators[model_name] = ClassifierEvaluator(model_name, y_test, y_pred)

    # Ensemble evaluator
    evaluators['SecurityFirstEnsemble'] = ClassifierEvaluator('SecurityFirstEnsemble', y_test, y_pred_ensemble)

    # Check if we should create save paths for visualizations
    save_paths = {}
    if args.save_plots_dir:
        import os
        os.makedirs(args.save_plots_dir, exist_ok=True)
        save_paths = {
            'metrics_comparison': os.path.join(args.save_plots_dir, 'model_comparison.png')
        }

    # Plot metrics comparison if requested
    if args.plot_metrics_comparison and evaluators:
        print("\n" + "="*70)
        print("PLOTTING METRICS COMPARISON")
        print("="*70)

        # Get ensemble evaluator
        ensemble_evaluator = evaluators['SecurityFirstEnsemble']

        # Create comparison with first individual model
        individual_models = [m for m in evaluators.keys() if m != 'SecurityFirstEnsemble']
        if individual_models:
            # Plot metrics comparison with the first individual model
            ensemble_evaluator.plot_metrics_comparison(
                other_evaluator=evaluators[individual_models[0]],
                save_path=save_paths.get('metrics_comparison')
            )

    # Plot confusion matrices if requested
    if args.plot_confusion_matrices:
        print("\n" + "="*70)
        print("PLOTTING CONFUSION MATRICES")
        print("="*70)

        # Plot confusion matrix for ensemble
        print(f"\nConfusion matrix for SecurityFirstEnsemble:")
        evaluators['SecurityFirstEnsemble'].plot_confusion_matrix()

        # Plot confusion matrices for individual models
        for model_name in individual_results.keys():
            if model_name != 'SecurityFirstEnsemble':
                print(f"\nConfusion matrix for {model_name}:")
                evaluators[model_name].plot_confusion_matrix()

    # Compare ensemble with each individual model
    print(f"\n{'='*70}")
    print("COMPARISON RESULTS")
    print(f"{'='*70}")

    for model_name in [m for m in individual_results.keys() if m != 'SecurityFirstEnsemble']:
        print(f"\nComparing SecurityFirstEnsemble vs {model_name}:")
        comparison = evaluators['SecurityFirstEnsemble'].compare_with_other(
            evaluators[model_name],
            verbose=True
        )

    # Save comparison results if requested
    if args.save_results:
        results_path = Path(args.save_results)

        comparison_data = []
        for model_name in individual_results.keys():
            if model_name != 'SecurityFirstEnsemble':
                comparison = evaluators['SecurityFirstEnsemble'].compare_with_other(
                    evaluators[model_name],
                    verbose=False
                )
                comparison_data.append({
                    'model': model_name,
                    'comparison': comparison
                })

        with open(results_path, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        print(f"\nComparison results saved to: {results_path}")

    return evaluators


def quick_evaluate_command(args):
    """Handle the quick-evaluate command."""
    print(f"\n{'='*70}")
    print("QUICK EVALUATION")
    print(f"{'='*70}")

    # Load data
    X, y = load_fraud_data(args.transactions, args.identity)

    if args.model:
        # Load model
        with open(args.model, 'rb') as f:
            model = pickle.load(f)

        # Make predictions
        y_pred = model.predict(X)

        # Get predicted probabilities if model supports it
        y_pred_proba = None
        try:
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X)[:, 1]
        except Exception as e:
            print(f"Note: Could not get predicted probabilities: {e}")

        # Create evaluator
        evaluator = ClassifierEvaluator(
            classifier_name=args.model_name if args.model_name else "LoadedModel",
            y_true=y,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba
        )

        # Check if we should create save paths
        save_paths = {}
        if args.save_plots_dir:
            import os
            os.makedirs(args.save_plots_dir, exist_ok=True)
            model_name_safe = args.model_name if args.model_name else "model"
            save_paths = {
                'confusion_matrix': os.path.join(args.save_plots_dir, f'{model_name_safe}_confusion_matrix.png'),
                'roc_curve': os.path.join(args.save_plots_dir, f'{model_name_safe}_roc_curve.png'),
                'pr_curve': os.path.join(args.save_plots_dir, f'{model_name_safe}_pr_curve.png'),
                'threshold_analysis': os.path.join(args.save_plots_dir, f'{model_name_safe}_threshold_analysis.png')
            }

        # Plot confusion matrix if requested
        if args.plot_confusion_matrix:
            print("\n" + "="*70)
            print("PLOTTING CONFUSION MATRIX")
            print("="*70)
            evaluator.plot_confusion_matrix(
                save_path=save_paths.get('confusion_matrix'),
                normalize=args.normalize_cm
            )

        # Evaluate with visualization options
        results = evaluator.evaluate(
            verbose=True,
            include_confusion_matrix=args.confusion_matrix,
            plot_confusion_matrix=False,  # Already handled above
            plot_roc_curve=args.plot_roc_curve,
            plot_pr_curve=args.plot_pr_curve,
            plot_threshold_analysis=args.plot_threshold_analysis
        )

        # Additional visualizations not in evaluate method
        if args.plot_roc_curve and y_pred_proba is not None:
            print("\n" + "="*70)
            print("PLOTTING ROC CURVE")
            print("="*70)
            evaluator.plot_roc_curve(save_path=save_paths.get('roc_curve'))

        if args.plot_pr_curve and y_pred_proba is not None:
            print("\n" + "="*70)
            print("PLOTTING PRECISION-RECALL CURVE")
            print("="*70)
            evaluator.plot_precision_recall_curve(save_path=save_paths.get('pr_curve'))

        if args.plot_threshold_analysis and y_pred_proba is not None:
            print("\n" + "="*70)
            print("PLOTTING THRESHOLD ANALYSIS")
            print("="*70)
            evaluator.plot_threshold_analysis(save_path=save_paths.get('threshold_analysis'))

        # Generate all visualizations if save directory is specified
        if args.save_plots_dir and y_pred_proba is not None:
            print("\n" + "="*70)
            print("GENERATING ALL VISUALIZATIONS")
            print("="*70)
            evaluator.visualize_all(save_dir=args.save_plots_dir)

        return results
    else:
        # Simple data statistics
        print(f"Data shape: {X.shape}")
        print(f"Fraud samples: {y.sum()} ({y.mean():.2%})")
        print(f"Legitimate samples: {(y == 0).sum()} ({(y == 0).mean():.2%})")

        # Basic feature stats
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(f"\nNumeric features: {len(numeric_cols)}")
            print("Sample statistics:")
            for col in numeric_cols[:5]:  # Show first 5 numeric columns
                print(f"  {col}: mean={X[col].mean():.2f}, std={X[col].std():.2f}")

        categorical_cols = X.select_dtypes(exclude=[np.number]).columns
        if len(categorical_cols) > 0:
            print(f"\nCategorical features: {len(categorical_cols)}")
            print("Sample unique values:")
            for col in categorical_cols[:3]:  # Show first 3 categorical columns
                print(f"  {col}: {X[col].nunique()} unique values")

    return None


def visualize_command(args):
    """Handle the visualize command."""
    print(f"\n{'='*70}")
    print("GENERATING MODEL VISUALIZATIONS")
    print(f"{'='*70}")

    # Load trained model
    print(f"Loading model from: {args.model}")
    with open(args.model, 'rb') as f:
        model = pickle.load(f)

    if not isinstance(model, SecurityFirstEnsemble):
        print(f"Warning: Loaded model is not a SecurityFirstEnsemble: {type(model)}")

    # Load data for visualization
    print(f"Loading visualization data...")
    X, y = load_fraud_data(args.transactions, args.identity)
    print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")

    # Make predictions
    y_pred = model.predict(X)

    # Get predicted probabilities if available
    y_pred_proba = None
    try:
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X)[:, 1]
            print(f"✓ Predicted probabilities available for ROC/PR curves")
    except Exception as e:
        print(f"Note: Could not get predicted probabilities: {e}")

    # Create evaluator
    evaluator = ClassifierEvaluator(
        classifier_name=type(model).__name__,
        y_true=y,
        y_pred=y_pred,
        y_pred_proba=y_pred_proba
    )

    # Create save directory
    import os
    os.makedirs(args.save_plots_dir, exist_ok=True)
    print(f"\nSaving plots to: {args.save_plots_dir}")

    # Determine which plots to generate
    plots_to_generate = args.plots
    if 'all' in plots_to_generate:
        plots_to_generate = ['confusion_matrix', 'roc_curve', 'pr_curve',
                            'threshold_analysis', 'metrics_comparison', 'feature_importance']

    # Generate requested plots
    for plot_type in plots_to_generate:
        if plot_type == 'confusion_matrix':
            print(f"\n{'='*70}")
            print("GENERATING CONFUSION MATRIX")
            print(f"{'='*70}")
            save_path = os.path.join(args.save_plots_dir, 'confusion_matrix.png')
            evaluator.plot_confusion_matrix(
                save_path=save_path,
                normalize=args.normalize_cm
            )

        elif plot_type == 'roc_curve':
            if y_pred_proba is None:
                print(f"\nSkipping ROC curve: No predicted probabilities available")
                continue
            print(f"\n{'='*70}")
            print("GENERATING ROC CURVE")
            print(f"{'='*70}")
            save_path = os.path.join(args.save_plots_dir, 'roc_curve.png')
            evaluator.plot_roc_curve(save_path=save_path)

        elif plot_type == 'pr_curve':
            if y_pred_proba is None:
                print(f"\nSkipping Precision-Recall curve: No predicted probabilities available")
                continue
            print(f"\n{'='*70}")
            print("GENERATING PRECISION-RECALL CURVE")
            print(f"{'='*70}")
            save_path = os.path.join(args.save_plots_dir, 'pr_curve.png')
            evaluator.plot_precision_recall_curve(save_path=save_path)

        elif plot_type == 'threshold_analysis':
            if y_pred_proba is None:
                print(f"\nSkipping threshold analysis: No predicted probabilities available")
                continue
            print(f"\n{'='*70}")
            print("GENERATING THRESHOLD ANALYSIS")
            print(f"{'='*70}")
            save_path = os.path.join(args.save_plots_dir, 'threshold_analysis.png')
            evaluator.plot_threshold_analysis(save_path=save_path)

        elif plot_type == 'feature_importance':
            # Check if model supports feature importance
            if not (hasattr(model, 'feature_importances_') or hasattr(model, 'coef_')):
                print(f"\nSkipping feature importance: Model doesn't support feature importance")
                continue

            print(f"\n{'='*70}")
            print("GENERATING FEATURE IMPORTANCE")
            print(f"{'='*70}")

            # Load feature names if provided
            feature_names = None
            if args.feature_names:
                try:
                    feature_df = pd.read_csv(args.feature_names)
                    if 'feature_name' in feature_df.columns:
                        feature_names = feature_df['feature_name'].tolist()
                        print(f"Loaded {len(feature_names)} feature names")
                except Exception as e:
                    print(f"Warning: Could not load feature names: {e}")

            save_path = os.path.join(args.save_plots_dir, 'feature_importance.png')
            evaluator.plot_feature_importance(
                model=model,
                feature_names=feature_names,
                top_n=args.top_features,
                save_path=save_path
            )

        elif plot_type == 'metrics_comparison':
            print(f"\n{'='*70}")
            print("GENERATING METRICS COMPARISON")
            print(f"{'='*70}")
            print("Note: Metrics comparison requires another model for comparison.")
            print("Use the 'compare' command for model comparisons.")

    print(f"\n{'='*70}")
    print("VISUALIZATION COMPLETED")
    print(f"{'='*70}")
    print(f"All requested plots saved to: {args.save_plots_dir}")

    return evaluator


def interactive_command(args):
    """Handle the interactive command-line interface."""
    print(f"\n{'='*70}")
    print("FRAUD DETECTION INTERACTIVE CLI")
    print(f"{'='*70}")
    print("Entering interactive mode... Type 'help' for available commands.")

    # Initialize state
    training_data = None
    test_data = None
    loaded_model = None
    trained_ensemble = None

    # Command handlers
    command_handlers = {
        'help': lambda: print_help(),
        'load-train': lambda: load_training_data_interactive(),
        'load-test': lambda: load_test_data_interactive(),
        'train': lambda: train_model_interactive(),
        'save': lambda: save_model_interactive(),
        'load': lambda: load_model_interactive(),
        'test': lambda: test_model_interactive(),
        'status': lambda: show_status(),
        'clear': lambda: clear_screen(),
        'exit': lambda: exit_interactive()
    }

    def print_help():
        """Print available commands."""
        print("\nAvailable Commands:")
        print("  help                    - Show this help message")
        print("  load-train              - Load training dataset")
        print("  load-test               - Load testing dataset")
        print("  train                   - Train a new model")
        print("  save <path>             - Save trained model to path")
        print("  load <path>             - Load model from path")
        print("  test                    - Test loaded/trained model")
        print("  visualize               - Generate visualizations")
        print("  status                  - Show current state")
        print("  clear                   - Clear screen")
        print("  exit                    - Exit interactive mode")
        print("\nExamples:")
        print("  load-train              # Load training data")
        print("  train --voting hard     # Train with hard voting")
        print("  save model.pkl          # Save model to model.pkl")
        print("  load saved_model.pkl    # Load model from file")
        print("  visualize               # Generate visualizations")

    def load_training_data_interactive():
        """Load training data interactively."""
        nonlocal training_data
        print("\nLoad Training Dataset")
        print("-" * 40)

        if args.transactions and args.identity:
            # Use command-line arguments if provided
            transactions_path = args.transactions
            identity_path = args.identity
            print(f"Using provided paths:")
            print(f"  Transactions: {transactions_path}")
            print(f"  Identity: {identity_path}")
        else:
            # Prompt for paths
            transactions_path = input("Enter transactions CSV path: ").strip()
            identity_path = input("Enter identity CSV path: ").strip()

        try:
            X, y = load_fraud_data(transactions_path, identity_path)
            training_data = {'X': X, 'y': y, 'transactions_path': transactions_path, 'identity_path': identity_path}
            print(f"✓ Training data loaded: {X.shape[0]} samples, {X.shape[1]} features")
            print(f"  Fraud rate: {y.sum()}/{len(y)} ({y.mean():.2%})")
        except Exception as e:
            print(f"✗ Error loading training data: {e}")

    def load_test_data_interactive():
        """Load testing data interactively."""
        nonlocal test_data
        print("\nLoad Testing Dataset")
        print("-" * 40)

        if args.transactions and args.identity:
            # Use command-line arguments if provided
            transactions_path = args.transactions
            identity_path = args.identity
            print(f"Using provided paths:")
            print(f"  Transactions: {transactions_path}")
            print(f"  Identity: {identity_path}")
        else:
            # Prompt for paths
            transactions_path = input("Enter transactions CSV path: ").strip()
            identity_path = input("Enter identity CSV path: ").strip()

        try:
            X, y = load_fraud_data(transactions_path, identity_path)
            test_data = {'X': X, 'y': y, 'transactions_path': transactions_path, 'identity_path': identity_path}
            print(f"✓ Test data loaded: {X.shape[0]} samples, {X.shape[1]} features")
            print(f"  Fraud rate: {y.sum()}/{len(y)} ({y.mean():.2%})")
        except Exception as e:
            print(f"✗ Error loading test data: {e}")

    def train_model_interactive():
        """Train a model interactively."""
        nonlocal training_data, trained_ensemble

        if training_data is None:
            print("✗ No training data loaded. Use 'load-train' first.")
            return

        print("\nTrain Model")
        print("-" * 40)

        # Get training parameters
        print("Training Parameters:")
        voting_type = input("Voting type (hard/soft/stacked) [hard]: ").strip() or "hard"
        tie_breaker = input("Tie breaker (malware/reject/confidence) [malware]: ").strip() or "malware"
        sample_fraction_input = input("Sample fraction (0.0-1.0) [1.0]: ").strip() or "1.0"

        try:
            sample_fraction = float(sample_fraction_input)
        except ValueError:
            print("✗ Invalid sample fraction. Using 1.0")
            sample_fraction = 1.0

        print(f"\nTraining with {voting_type} voting, tie-breaker: {tie_breaker}")
        if sample_fraction < 1.0:
            print(f"Using {sample_fraction:.0%} of data for training")

        try:
            X_train = training_data['X']
            y_train = training_data['y']

            # Apply sampling if needed
            if sample_fraction < 1.0:
                n_samples = int(len(X_train) * sample_fraction)
                indices = np.random.choice(len(X_train), n_samples, replace=False)
                X_train = X_train.iloc[indices].copy()
                y_train = y_train.iloc[indices].copy()

            # Create and train ensemble
            ensemble = SecurityFirstEnsemble(
                tie_breaker=tie_breaker,
                voting_type=voting_type
            )

            print("\n" + "="*70)
            print(f"TRAINING SECURITY-FIRST ENSEMBLE")
            print(f"Voting: {voting_type} | Tie-breaker: {tie_breaker}")
            print("="*70)

            ensemble.fit(X_train, y_train)
            trained_ensemble = ensemble
            # Clear any loaded model
            loaded_model = None

            print(f"\n✓ Model trained successfully!")
            print(f"  Model type: SecurityFirstEnsemble")
            print(f"  Voting: {voting_type}")
            print(f"  Tie-breaker: {tie_breaker}")

        except Exception as e:
            print(f"✗ Error training model: {e}")
            import traceback
            traceback.print_exc()

    def save_model_interactive():
        """Save a model interactively."""
        nonlocal trained_ensemble, loaded_model

        model = trained_ensemble or loaded_model
        if model is None:
            print("✗ No model available to save. Train or load a model first.")
            return

        print("\nSave Model")
        print("-" * 40)

        save_path = input("Enter save path (e.g., model.pkl): ").strip()
        if not save_path:
            print("✗ No save path provided.")
            return

        try:
            with open(save_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"✓ Model saved to: {save_path}")
        except Exception as e:
            print(f"✗ Error saving model: {e}")

    def load_model_interactive():
        """Load a model interactively."""
        nonlocal loaded_model, trained_ensemble

        print("\nLoad Model")
        print("-" * 40)

        load_path = input("Enter model path to load: ").strip()
        if not load_path:
            print("✗ No model path provided.")
            return

        try:
            with open(load_path, 'rb') as f:
                model = pickle.load(f)

            if not isinstance(model, SecurityFirstEnsemble):
                print(f"⚠ Warning: Loaded model is not a SecurityFirstEnsemble: {type(model)}")

            loaded_model = model
            trained_ensemble = None  # Clear any trained model

            print(f"✓ Model loaded from: {load_path}")
            print(f"  Model type: {type(model).__name__}")

            # Try to get voting type if it's a SecurityFirstEnsemble
            if hasattr(model, 'voting_type'):
                print(f"  Voting: {model.voting_type}")
            if hasattr(model, 'tie_breaker'):
                print(f"  Tie-breaker: {model.tie_breaker}")

        except FileNotFoundError:
            print(f"✗ File not found: {load_path}")
        except Exception as e:
            print(f"✗ Error loading model: {e}")

    def test_model_interactive():
        """Test a model interactively."""
        nonlocal trained_ensemble, loaded_model, test_data

        model = trained_ensemble or loaded_model
        if model is None:
            print("✗ No model available to test. Train or load a model first.")
            return

        if test_data is None:
            print("✗ No test data loaded. Use 'load-test' first.")
            return

        print("\nTest Model")
        print("-" * 40)

        threshold_input = input("Probability threshold (0.0-1.0) [0.5]: ").strip() or "0.5"

        try:
            threshold = float(threshold_input)
        except ValueError:
            print("✗ Invalid threshold. Using 0.5")
            threshold = 0.5

        print(f"\nTesting model on {test_data['X'].shape[0]} samples...")
        print(f"Threshold: {threshold}")

        try:
            X_test = test_data['X']
            y_test = test_data['y']

            # Evaluate the model
            results = model.evaluate(
                X_test=X_test,
                y_test=y_test,
                verbose=True,
                threshold=threshold
            )

            print(f"\n✓ Testing completed successfully!")

            # Ask if user wants to generate visualizations
            generate_viz = input("\nGenerate visualizations? (yes/no) [no]: ").strip().lower()
            if generate_viz in ['yes', 'y']:
                visualize_current_test(model, X_test, y_test, results)

        except Exception as e:
            print(f"✗ Error testing model: {e}")
            import traceback
            traceback.print_exc()

    def visualize_current_test(model, X_test, y_test, results):
        """Generate visualizations for current test."""
        print(f"\nVisualization Options:")
        print("  1. Confusion Matrix")
        print("  2. ROC Curve (if probabilities available)")
        print("  3. Precision-Recall Curve (if probabilities available)")
        print("  4. Threshold Analysis (if probabilities available)")
        print("  5. Feature Importance (if model supports it)")
        print("  6. All visualizations")

        choice = input("\nChoose visualization (1-6) or 'cancel': ").strip()
        if choice == 'cancel':
            return

        # Get predicted probabilities if available
        y_pred_proba = None
        try:
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
        except Exception as e:
            print(f"Note: Could not get predicted probabilities: {e}")

        # Create evaluator
        evaluator = ClassifierEvaluator(
            classifier_name=type(model).__name__,
            y_true=y_test,
            y_pred=model.predict(X_test),
            y_pred_proba=y_pred_proba
        )

        # Get save directory
        save_dir = input("\nSave directory for plots (leave empty to skip saving): ").strip()
        save_paths = {}
        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            model_name = type(model).__name__
            save_paths = {
                'confusion_matrix': os.path.join(save_dir, f'{model_name}_confusion_matrix.png'),
                'roc_curve': os.path.join(save_dir, f'{model_name}_roc_curve.png'),
                'pr_curve': os.path.join(save_dir, f'{model_name}_pr_curve.png'),
                'threshold_analysis': os.path.join(save_dir, f'{model_name}_threshold_analysis.png'),
                'feature_importance': os.path.join(save_dir, f'{model_name}_feature_importance.png')
            }

        # Generate selected visualizations
        if choice == '1' or choice == '6':
            print(f"\n{'='*70}")
            print("GENERATING CONFUSION MATRIX")
            print(f"{'='*70}")
            normalize = input("Normalize confusion matrix? (yes/no) [no]: ").strip().lower() in ['yes', 'y']
            evaluator.plot_confusion_matrix(save_path=save_paths.get('confusion_matrix'), normalize=normalize)

        if (choice == '2' or choice == '6') and y_pred_proba is not None:
            print(f"\n{'='*70}")
            print("GENERATING ROC CURVE")
            print(f"{'='*70}")
            evaluator.plot_roc_curve(save_path=save_paths.get('roc_curve'))

        if (choice == '3' or choice == '6') and y_pred_proba is not None:
            print(f"\n{'='*70}")
            print("GENERATING PRECISION-RECALL CURVE")
            print(f"{'='*70}")
            evaluator.plot_precision_recall_curve(save_path=save_paths.get('pr_curve'))

        if (choice == '4' or choice == '6') and y_pred_proba is not None:
            print(f"\n{'='*70}")
            print("GENERATING THRESHOLD ANALYSIS")
            print(f"{'='*70}")
            evaluator.plot_threshold_analysis(save_path=save_paths.get('threshold_analysis'))

        if (choice == '5' or choice == '6') and (hasattr(model, 'feature_importances_') or hasattr(model, 'coef_')):
            print(f"\n{'='*70}")
            print("GENERATING FEATURE IMPORTANCE")
            print(f"{'='*70}")

            # Try to get feature names from test data
            feature_names = None
            if hasattr(X_test, 'columns'):
                feature_names = list(X_test.columns)
                print(f"Using {len(feature_names)} feature names from data")

            evaluator.plot_feature_importance(
                model=model,
                feature_names=feature_names,
                top_n=20,
                save_path=save_paths.get('feature_importance')
            )

        if choice == '6' and y_pred_proba is not None:
            print(f"\n{'='*70}")
            print("GENERATING ALL VISUALIZATIONS")
            print(f"{'='*70}")
            evaluator.visualize_all(save_dir=save_dir)

        if save_dir:
            print(f"\n✓ Visualizations saved to: {save_dir}")

    def visualize_model_interactive():
        """Visualize a model interactively."""
        nonlocal trained_ensemble, loaded_model, test_data

        model = trained_ensemble or loaded_model
        if model is None:
            print("✗ No model available to visualize. Train or load a model first.")
            return

        if test_data is None:
            print("✗ No test data loaded. Use 'load-test' first.")
            return

        print("\nModel Visualization")
        print("-" * 40)

        print(f"\nUsing test data with {test_data['X'].shape[0]} samples")
        X_test = test_data['X']
        y_test = test_data['y']

        # Call the visualization helper
        visualize_current_test(model, X_test, y_test, None)

    def show_status():
        """Show current interactive session status."""
        print("\nCurrent Status")
        print("-" * 40)

        # Training data status
        if training_data:
            print(f"✓ Training Data: Loaded")
            print(f"  Samples: {training_data['X'].shape[0]}")
            print(f"  Features: {training_data['X'].shape[1]}")
            print(f"  Paths: {training_data['transactions_path']}, {training_data['identity_path']}")
        else:
            print(f"✗ Training Data: Not loaded")

        # Test data status
        if test_data:
            print(f"✓ Test Data: Loaded")
            print(f"  Samples: {test_data['X'].shape[0]}")
            print(f"  Features: {test_data['X'].shape[1]}")
            print(f"  Paths: {test_data['transactions_path']}, {test_data['identity_path']}")
        else:
            print(f"✗ Test Data: Not loaded")

        # Model status
        if trained_ensemble:
            print(f"✓ Model: Trained (not saved)")
            print(f"  Type: SecurityFirstEnsemble")
            if hasattr(trained_ensemble, 'voting_type'):
                print(f"  Voting: {trained_ensemble.voting_type}")
            if hasattr(trained_ensemble, 'tie_breaker'):
                print(f"  Tie-breaker: {trained_ensemble.tie_breaker}")
        elif loaded_model:
            print(f"✓ Model: Loaded from file")
            print(f"  Type: {type(loaded_model).__name__}")
            if hasattr(loaded_model, 'voting_type'):
                print(f"  Voting: {loaded_model.voting_type}")
            if hasattr(loaded_model, 'tie_breaker'):
                print(f"  Tie-breaker: {loaded_model.tie_breaker}")
        else:
            print(f"✗ Model: No model available")

    def clear_screen():
        """Clear the terminal screen."""
        import os
        os.system('cls' if os.name == 'nt' else 'clear')
        print("Screen cleared.")

    def exit_interactive():
        """Exit interactive mode."""
        print("\nExiting interactive mode...")
        return True

    # Main interactive loop
    while True:
        try:
            print("\n" + "-" * 50)
            command_input = input("fraud-cli> ").strip()

            if not command_input:
                continue

            # Split command and arguments
            parts = command_input.split()
            cmd = parts[0].lower()
            cmd_args = parts[1:] if len(parts) > 1 else []

            if cmd == 'save' and cmd_args:
                # Special handling for save command with path
                save_path = cmd_args[0]
                model = trained_ensemble or loaded_model
                if model is None:
                    print("✗ No model available to save. Train or load a model first.")
                    continue
                try:
                    with open(save_path, 'wb') as f:
                        pickle.dump(model, f)
                    print(f"✓ Model saved to: {save_path}")
                except Exception as e:
                    print(f"✗ Error saving model: {e}")
                continue
            elif cmd == 'load' and cmd_args:
                # Special handling for load command with path
                load_path = cmd_args[0]
                try:
                    with open(load_path, 'rb') as f:
                        model = pickle.load(f)

                    if not isinstance(model, SecurityFirstEnsemble):
                        print(f"⚠ Warning: Loaded model is not a SecurityFirstEnsemble: {type(model)}")

                    loaded_model = model
                    trained_ensemble = None
                    print(f"✓ Model loaded from: {load_path}")
                    print(f"  Model type: {type(model).__name__}")

                except FileNotFoundError:
                    print(f"✗ File not found: {load_path}")
                except Exception as e:
                    print(f"✗ Error loading model: {e}")
                continue

            # Handle other commands
            if cmd in command_handlers:
                should_exit = command_handlers[cmd]()
                if should_exit:
                    break
            else:
                print(f"✗ Unknown command: {cmd}. Type 'help' for available commands.")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'exit' to quit or continue with commands.")
        except EOFError:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\n✗ Error: {e}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="CLI for Fraud Detection Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train an ensemble with hard voting
  python fraud_cli.py train --transactions data/trans.csv --identity data/id.csv

  # Test a saved model
  python fraud_cli.py test --model model.pkl --transactions test.csv --identity test_id.csv

  # Compare ensemble vs individual models
  python fraud_cli.py compare --transactions data.csv --identity id.csv

  # Quick evaluation of data or model
  python fraud_cli.py quick-evaluate --transactions data.csv --identity id.csv --model model.pkl

  # Generate visualizations for a model
  python fraud_cli.py visualize --model model.pkl --transactions data.csv --identity id.csv --save-plots-dir plots/

  # Interactive mode
  python fraud_cli.py interactive
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute", required=True)

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a new ensemble model")
    train_parser.add_argument("--transactions", required=True, help="Path to transactions CSV file")
    train_parser.add_argument("--identity", required=True, help="Path to identity CSV file")
    train_parser.add_argument("--voting", choices=["hard", "soft", "stacked"], default="hard",
                             help="Voting type for ensemble")
    train_parser.add_argument("--tie-breaker", choices=["malware", "reject", "confidence"],
                             default="malware", help="Tie-breaking strategy")
    train_parser.add_argument("--dataset-name", default="fraud_data",
                             help="Name for the dataset sample")
    train_parser.add_argument("--save-model", help="Path to save the trained model (optional)")
    train_parser.add_argument("--save-results", help="Path to save training results (optional)")
    train_parser.add_argument("--sample-fraction", type=float, default=1.0,
                             help="Fraction of data to use for training (0.0-1.0)")
    train_parser.add_argument("--plot-training-metrics", action="store_true",
                            help="Plot training metrics and visualizations")
    train_parser.add_argument("--save-plots-dir", help="Directory to save plot images (optional)")
    train_parser.set_defaults(func=train_command)

    # Test command
    test_parser = subparsers.add_parser("test", help="Test a trained model")
    test_parser.add_argument("--model", required=True, help="Path to trained model file")
    test_parser.add_argument("--transactions", required=True, help="Path to test transactions CSV file")
    test_parser.add_argument("--identity", required=True, help="Path to test identity CSV file")
    test_parser.add_argument("--threshold", type=float, default=0.5,
                            help="Probability threshold for classification (for soft voting)")
    test_parser.add_argument("--save-results", help="Path to save test results (optional)")
    test_parser.add_argument("--plot-confusion-matrix", action="store_true",
                            help="Plot confusion matrix (requires matplotlib)")
    test_parser.add_argument("--plot-roc-curve", action="store_true",
                            help="Plot ROC curve (requires predicted probabilities)")
    test_parser.add_argument("--plot-pr-curve", action="store_true",
                            help="Plot Precision-Recall curve (requires predicted probabilities)")
    test_parser.add_argument("--plot-threshold-analysis", action="store_true",
                            help="Plot threshold analysis (requires predicted probabilities)")
    test_parser.add_argument("--normalize-cm", action="store_true",
                            help="Normalize confusion matrix")
    test_parser.add_argument("--save-plots-dir", help="Directory to save plot images (optional)")
    test_parser.set_defaults(func=test_command)

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare ensemble vs individual models")
    compare_parser.add_argument("--transactions", required=True, help="Path to transactions CSV file")
    compare_parser.add_argument("--identity", required=True, help="Path to identity CSV file")
    compare_parser.add_argument("--voting", choices=["hard", "soft", "stacked"], default="hard",
                              help="Voting type for ensemble")
    compare_parser.add_argument("--tie-breaker", choices=["malware", "reject", "confidence"],
                              default="malware", help="Tie-breaking strategy")
    compare_parser.add_argument("--dataset-name", default="fraud_data",
                              help="Name for the dataset sample")
    compare_parser.add_argument("--save-results", help="Path to save comparison results (optional)")
    compare_parser.add_argument("--plot-metrics-comparison", action="store_true",
                              help="Plot metrics comparison between models")
    compare_parser.add_argument("--plot-confusion-matrices", action="store_true",
                              help="Plot confusion matrices for all models")
    compare_parser.add_argument("--save-plots-dir", help="Directory to save plot images (optional)")
    compare_parser.set_defaults(func=compare_command)

    # Quick evaluate command
    quick_eval_parser = subparsers.add_parser("quick-evaluate",
                                              help="Quick evaluation of data or model")
    quick_eval_parser.add_argument("--transactions", required=True, help="Path to transactions CSV file")
    quick_eval_parser.add_argument("--identity", required=True, help="Path to identity CSV file")
    quick_eval_parser.add_argument("--model", help="Path to model file for evaluation (optional)")
    quick_eval_parser.add_argument("--model-name", default="EnsembleModel",
                                  help="Name to use for the model in output")
    quick_eval_parser.add_argument("--confusion-matrix", action="store_true",
                                  help="Include confusion matrix in output")
    quick_eval_parser.add_argument("--plot-confusion-matrix", action="store_true",
                                  help="Plot confusion matrix (requires matplotlib)")
    quick_eval_parser.add_argument("--plot-roc-curve", action="store_true",
                                  help="Plot ROC curve (requires predicted probabilities)")
    quick_eval_parser.add_argument("--plot-pr-curve", action="store_true",
                                  help="Plot Precision-Recall curve (requires predicted probabilities)")
    quick_eval_parser.add_argument("--plot-threshold-analysis", action="store_true",
                                  help="Plot threshold analysis (requires predicted probabilities)")
    quick_eval_parser.add_argument("--normalize-cm", action="store_true",
                                  help="Normalize confusion matrix")
    quick_eval_parser.add_argument("--save-plots-dir", help="Directory to save plot images (optional)")
    quick_eval_parser.set_defaults(func=quick_evaluate_command)

    # Visualize command
    visualize_parser = subparsers.add_parser("visualize",
                                            help="Generate visualizations for a trained model")
    visualize_parser.add_argument("--model", required=True, help="Path to trained model file")
    visualize_parser.add_argument("--transactions", required=True, help="Path to transactions CSV file for visualization")
    visualize_parser.add_argument("--identity", required=True, help="Path to identity CSV file for visualization")
    visualize_parser.add_argument("--save-plots-dir", required=True,
                                 help="Directory to save plot images")
    visualize_parser.add_argument("--plots", nargs='+', default=['all'],
                                 choices=['all', 'confusion_matrix', 'roc_curve', 'pr_curve',
                                         'threshold_analysis', 'metrics_comparison', 'feature_importance'],
                                 help="Types of plots to generate (default: all)")
    visualize_parser.add_argument("--normalize-cm", action="store_true",
                                 help="Normalize confusion matrix")
    visualize_parser.add_argument("--feature-names", help="CSV file with feature names (optional)")
    visualize_parser.add_argument("--top-features", type=int, default=20,
                                 help="Number of top features to show in feature importance plot")
    visualize_parser.set_defaults(func=visualize_command)

    # Interactive command
    interactive_parser = subparsers.add_parser("interactive",
                                              help="Interactive command-line interface")
    interactive_parser.add_argument("--transactions", help="Path to transactions CSV file (optional, can be loaded later)")
    interactive_parser.add_argument("--identity", help="Path to identity CSV file (optional, can be loaded later)")
    interactive_parser.set_defaults(func=interactive_command)

    # Parse arguments
    args = parser.parse_args()

    try:
        # Execute the command
        result = args.func(args)
        print(f"\nCommand '{args.command}' completed successfully")

    except FileNotFoundError as e:
        print(f"\nError: File not found - {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nError during execution: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()