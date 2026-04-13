#!/usr/bin/env python3
"""
Drift-based retraining system for malware detection.
Integrates with existing drift detection and model evaluation infrastructure.
"""

import numpy as np
import pandas as pd
import json
import pickle
import os
from datetime import datetime
from typing import Dict, Tuple, List, Optional, Any, Union
from pathlib import Path

# Import existing components
from drift import DriftAnalyzer
from dataset import Dataset
from sample import Sample
from models import (ClassifierEvaluator, train_and_evaluate_classifiers,
                     quick_evaluate_classifier, SecurityFirstEnsemble,
                     train_and_evaluate_ensemble)


class ModelRegistry:
    """
    Manages versioned models with metadata and performance tracking.
    """

    def __init__(self, registry_path: str = "model_registry"):
        """
        Initialize model registry.

        Args:
            registry_path: Path to store model files and metadata
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(exist_ok=True, parents=True)
        self.metadata_file = self.registry_path / "metadata.json"
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """Load or create metadata file."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {"models": [], "current_version": None}

    def _save_metadata(self):
        """Save metadata to file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def register_model(self, model: Any, model_name: str,
                      version: str, performance: Dict[str, float],
                      training_years: List[str], validation_year: str,
                      features: List[str], retraining_reason: str = "periodic"):
        """
        Register a new model version.

        Args:
            model: The trained model object
            model_name: Name of the model (e.g., "RandomForest", "LogisticRegression")
            version: Version identifier (e.g., "v1.0.0", "2025-01-15")
            performance: Dictionary of performance metrics
            training_years: List of years used for training
            validation_year: Year used for validation
            features: List of feature names used
            retraining_reason: Reason for retraining (periodic, drift_detected, etc.)
        """
        # Save model to file
        model_filename = f"{model_name}_{version}.pkl"
        model_path = self.registry_path / model_filename

        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        # Create metadata entry
        model_entry = {
            "model_name": model_name,
            "version": version,
            "filename": model_filename,
            "registration_date": datetime.now().isoformat(),
            "performance": performance,
            "training_years": training_years,
            "validation_year": validation_year,
            "feature_count": len(features),
            "retraining_reason": retraining_reason,
            "is_active": False  # Will be activated by set_current_version
        }

        # Add to metadata
        self.metadata["models"].append(model_entry)
        self._save_metadata()

        print(f"Registered {model_name} version {version}")
        return model_entry

    def set_current_version(self, model_name: str, version: str):
        """Set a specific model version as the current active version."""
        # Deactivate all models
        for model in self.metadata["models"]:
            model["is_active"] = False

        # Activate specified model
        for model in self.metadata["models"]:
            if model["model_name"] == model_name and model["version"] == version:
                model["is_active"] = True
                self.metadata["current_version"] = f"{model_name}_{version}"
                print(f"Activated {model_name} version {version}")
                break

        self._save_metadata()

    def get_current_model(self, model_name: str) -> Optional[Tuple[Any, Dict]]:
        """Get the current active model and its metadata."""
        for model in self.metadata["models"]:
            if model["model_name"] == model_name and model.get("is_active", False):
                model_path = self.registry_path / model["filename"]
                with open(model_path, 'rb') as f:
                    loaded_model = pickle.load(f)
                return loaded_model, model

        return None

    def list_models(self) -> pd.DataFrame:
        """List all registered models as a DataFrame."""
        if not self.metadata["models"]:
            return pd.DataFrame()

        return pd.DataFrame(self.metadata["models"])

    def get_model_performance_history(self, model_name: str) -> pd.DataFrame:
        """Get performance history for a specific model."""
        history = []
        for model in self.metadata["models"]:
            if model["model_name"] == model_name:
                entry = {
                    "version": model["version"],
                    "registration_date": model["registration_date"],
                    "training_years": ", ".join(model["training_years"]),
                    "validation_year": model["validation_year"],
                    "retraining_reason": model["retraining_reason"],
                    "is_active": model["is_active"]
                }
                entry.update(model["performance"])
                history.append(entry)

        return pd.DataFrame(history)


class DriftAwareRetrainingSystem:
    """
    Main system for drift-based retraining of malware detection models.
    """

    def __init__(self,
                 drift_threshold: float = 0.3,  # 30% feature drift
                 performance_degradation_threshold: float = 0.05,  # 5% drop
                 registry_path: str = "model_registry"):
        """
        Initialize retraining system.

        Args:
            drift_threshold: Minimum drift rate to trigger retraining (0.0-1.0)
            performance_degradation_threshold: Minimum performance drop to trigger retraining
            registry_path: Path to model registry
        """
        self.drift_threshold = drift_threshold
        self.performance_threshold = performance_degradation_threshold
        self.registry = ModelRegistry(registry_path)
        self.drift_analyzer = None
        self.current_year = None

        # Track retraining history
        self.retraining_history = []

    def should_retrain_based_on_drift(self,
                                     dataset: Dataset,
                                     current_year: str,
                                     previous_year: str) -> Tuple[bool, float, pd.DataFrame]:
        """
        Check if concept drift warrants retraining.

        Args:
            dataset: Dataset containing samples from multiple years
            current_year: Year to evaluate (new data)
            previous_year: Previous year to compare against

        Returns:
            Tuple of (should_retrain, drift_rate, drift_details)
        """
        if self.drift_analyzer is None:
            self.drift_analyzer = DriftAnalyzer(dataset)

        try:
            # Analyze drift between consecutive years
            year_pairs = [(previous_year, current_year)]
            drift_df = self.drift_analyzer.analyze_year_pairs(year_pairs)

            if drift_df.empty:
                print(f"No common features between {previous_year} and {current_year}")
                return True, 1.0, pd.DataFrame()  # Force retraining if no common features

            # Calculate drift rate
            drift_rate = drift_df['drift_detected'].mean()

            # Generate summary
            summary = self.drift_analyzer.drift_summary(drift_df)

            print(f"Drift analysis {previous_year} -> {current_year}:")
            print(f"  Features analyzed: {len(drift_df)}")
            print(f"  Drift rate: {drift_rate:.2%}")
            print(f"  Drift threshold: {self.drift_threshold:.2%}")

            # Check if drift exceeds threshold
            should_retrain = drift_rate > self.drift_threshold

            if should_retrain:
                print(f"  Decision: RETRAIN (drift exceeds threshold)")
            else:
                print(f"  Decision: NO RETRAIN (drift below threshold)")

            return should_retrain, drift_rate, drift_df

        except Exception as e:
            print(f"Error in drift analysis: {e}")
            # If drift analysis fails, err on side of retraining
            return True, 1.0, pd.DataFrame()

    def should_retrain_based_on_performance(self,
                                          model_name: str,
                                          current_performance: Dict[str, float],
                                          validation_year: str) -> Tuple[bool, float]:
        """
        Check if performance degradation warrants retraining.

        Args:
            model_name: Name of the model to check
            current_performance: Performance on current year's data
            validation_year: Year used for validation

        Returns:
            Tuple of (should_retrain, performance_drop)
        """
        # Get model history
        history_df = self.registry.get_model_performance_history(model_name)

        if history_df.empty:
            print(f"No performance history for {model_name}, assuming retraining needed")
            return True, 1.0

        # Find the best historical performance
        best_recall = history_df['recall'].max()
        current_recall = current_performance.get('recall', 0)

        # Calculate performance drop
        if best_recall > 0:
            performance_drop = (best_recall - current_recall) / best_recall
        else:
            performance_drop = 1.0

        print(f"Performance analysis for {model_name}:")
        print(f"  Best historical Recall: {best_recall:.4f}")
        print(f"  Current Recall ({validation_year}): {current_recall:.4f}")
        print(f"  Performance drop: {performance_drop:.2%}")
        print(f"  Performance threshold: {self.performance_threshold:.2%}")

        should_retrain = performance_drop > self.performance_threshold

        if should_retrain:
            print(f"  Decision: RETRAIN (performance degradation exceeds threshold)")
        else:
            print(f"  Decision: NO RETRAIN (performance acceptable)")

        return should_retrain, performance_drop

    def progressive_validation_split(self,
                                   years: List[str],
                                   validation_year: str) -> Tuple[List[str], str]:
        """
        Create training/validation split using progressive validation strategy.

        Args:
            years: List of available years
            validation_year: Year to use for validation

        Returns:
            Tuple of (training_years, validation_year)
        """
        # Sort years chronologically
        sorted_years = sorted(years, key=int)

        if validation_year not in sorted_years:
            raise ValueError(f"Validation year {validation_year} not in available years")

        # Use all years before validation_year for training
        validation_idx = sorted_years.index(validation_year)
        training_years = sorted_years[:validation_idx]

        print(f"Progressive validation split:")
        print(f"  Training years: {training_years}")
        print(f"  Validation year: {validation_year}")

        return training_years, validation_year

    def prepare_training_data(self,
                            all_data: Dict[str, Tuple[pd.DataFrame, pd.Series]],
                            training_years: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Combine data from multiple years for training.

        Args:
            all_data: Dictionary of year -> (X, y)
            training_years: List of years to include in training

        Returns:
            Combined (X_train, y_train)
        """
        X_list = []
        y_list = []

        for year in training_years:
            if year in all_data:
                X_year, y_year = all_data[year]
                X_list.append(X_year)
                y_list.append(y_year)
                print(f"  Added {year}: {X_year.shape[0]} samples")
            else:
                print(f"  Warning: Year {year} not found in data")

        if not X_list:
            raise ValueError("No training data available")

        X_combined = pd.concat(X_list, ignore_index=True)
        y_combined = pd.concat(y_list, ignore_index=True)

        print(f"  Total training data: {X_combined.shape[0]} samples, {X_combined.shape[1]} features")

        return X_combined, y_combined

    def retrain_with_drift_awareness(self,
                                   all_data: Dict[str, Tuple[pd.DataFrame, pd.Series]],
                                   current_year: str,
                                   model_types: List[str] = ["LogisticRegression", "RandomForest"],
                                   force_retrain: bool = False) -> Dict[str, Any]:
        """
        Main retraining workflow with drift awareness.

        Args:
            all_data: All available data by year
            current_year: Current year to validate on
            model_types: List of model types to train
            force_retrain: Force retraining regardless of drift assessment

        Returns:
            Dictionary with retraining results
        """
        print("\n" + "="*70)
        print(f"DRIFT-AWARE RETRAINING SYSTEM - Year {current_year}")
        print("="*70)

        # Get available years
        available_years = list(all_data.keys())
        if current_year not in available_years:
            raise ValueError(f"Current year {current_year} not in available data")

        # Create dataset for drift analysis
        dataset = Dataset(all_data)

        # Find previous year for drift comparison
        sorted_years = sorted(available_years, key=int)
        current_idx = sorted_years.index(current_year)

        if current_idx == 0:
            print(f"First year {current_year}, no previous year for drift comparison")
            previous_year = None
            drift_based_retrain = True
            drift_rate = 1.0
            drift_details = pd.DataFrame()
        else:
            previous_year = sorted_years[current_idx - 1]

            # Check drift-based retraining need
            drift_based_retrain, drift_rate, drift_details = self.should_retrain_based_on_drift(
                dataset, current_year, previous_year
            )

        # Prepare training/validation split
        training_years, validation_year = self.progressive_validation_split(
            available_years, current_year
        )

        if not training_years:
            print(f"No historical data for training on {current_year}")
            return {"status": "failed", "reason": "no_training_data"}

        # Prepare training data
        X_train, y_train = self.prepare_training_data(all_data, training_years)

        # Get validation data
        X_val, y_val = all_data[validation_year]

        print(f"\nValidation data ({validation_year}): {X_val.shape[0]} samples")

        # Check performance of current model if available
        performance_based_retrain = False
        performance_drop = 0.0

        # Evaluate current models if they exist
        current_performances = {}
        for model_name in model_types:
            current_model = self.registry.get_current_model(model_name)
            if current_model:
                model, metadata = current_model
                # Evaluate on current year
                y_pred = model.predict(X_val)
                evaluator = ClassifierEvaluator(f"{model_name}_current", y_val, y_pred)
                current_perf = evaluator.calculate_metrics()
                current_performances[model_name] = current_perf

                # Check performance degradation
                should_retrain, drop = self.should_retrain_based_on_performance(
                    model_name, current_perf, validation_year
                )
                if should_retrain:
                    performance_based_retrain = True
                    performance_drop = max(performance_drop, drop)

        # Decision logic
        should_retrain = (
            force_retrain or
            drift_based_retrain or
            performance_based_retrain or
            not any(self.registry.get_current_model(m) for m in model_types)
        )

        print(f"\nRetraining Decision:")
        print(f"  Force retrain: {force_retrain}")
        print(f"  Drift-based retrain: {drift_based_retrain} (rate: {drift_rate:.2%})")
        print(f"  Performance-based retrain: {performance_based_retrain} (drop: {performance_drop:.2%})")
        print(f"  No current model: {not any(self.registry.get_current_model(m) for m in model_types)}")
        print(f"  FINAL DECISION: {'RETRAIN' if should_retrain else 'NO RETRAIN'}")

        if not should_retrain:
            print(f"\nSkipping retraining for {current_year}")
            return {
                "status": "skipped",
                "reason": "no_retraining_needed",
                "drift_rate": drift_rate,
                "performance_drop": performance_drop,
                "current_models": current_performances
            }

        # Perform retraining
        print(f"\nPerforming retraining for {current_year}...")

        # Train and evaluate classifiers
        results = train_and_evaluate_classifiers(
            X_train, X_val, y_train, y_val
        )

        # Register new models
        registered_models = {}
        retraining_reason = "periodic"
        if drift_based_retrain:
            retraining_reason = f"drift_detected_{drift_rate:.2%}"
        elif performance_based_retrain:
            retraining_reason = f"performance_degradation_{performance_drop:.2%}"
        elif force_retrain:
            retraining_reason = "forced"

        for model_name, result in results.items():
            if 'evaluation' in result:
                new_metrics = result['evaluation']['metrics']
                new_score = (new_metrics['recall'],
                             new_metrics['f1_score'],
                             new_metrics['precision'],
                             new_metrics['accuracy']
                            )
                old_metrics = current_performances.get(model_name)

                if old_metrics:
                    old_score = (old_metrics['recall'],
                                 old_metrics['f1_score'],
                                 old_metrics['precision'],
                                 old_metrics['accuracy'])
                    print(f"Comparing NEW ({new_score[0]:.4f} Recall) vs Current ({old_score[0]:.4f} Recall) on {current_year}")
                else:
                    # If no model exists for this type, new model wins by default.
                    old_score = (-1, -1, -1, -1)
                
                # Only register if the new model is better than the old model on prioritized metrics.
                if new_score >= old_score:
                    print ("Saving new model, it's performance is greater than the old model.")
                    model = result['model']
                    performance = result['evaluation']['metrics']
                    version = f"{current_year}_v{len(self.retraining_history) + 1}"

                    # Register model
                    model_entry = self.registry.register_model(
                        model=model,
                        model_name=model_name,
                        version=version,
                        performance=performance,
                        training_years=training_years,
                        validation_year=validation_year,
                        features=list(X_train.columns),
                        retraining_reason=retraining_reason
                    )

                    # Set as current version
                    self.registry.set_current_version(model_name, version)
                    registered_models[model_name] = model_entry


                else:
                    print("New model not saved. It's preformance is worse than the old model.")

        # Log retraining event
        retraining_event = {
            "date": datetime.now().isoformat(),
            "year": current_year,
            "drift_rate": drift_rate,
            "performance_drop": performance_drop,
            "retraining_reason": retraining_reason,
            "training_years": training_years,
            "validation_year": validation_year,
            "registered_models": list(registered_models.keys())
        }
        self.retraining_history.append(retraining_event)

        print(f"\nRetraining completed for {current_year}")
        print(f"Registered models: {list(registered_models.keys())}")

        return {
            "status": "success",
            "registered_models": registered_models,
            "retraining_event": retraining_event,
            "training_stats": {
                "training_samples": X_train.shape[0],
                "validation_samples": X_val.shape[0],
                "feature_count": X_train.shape[1]
            }
        }

    def run_progressive_validation_pipeline(self,
                                          all_data: Dict[str, Tuple[pd.DataFrame, pd.Series]],
                                          start_year: Optional[str] = None,
                                          end_year: Optional[str] = None,
                                          model_types: List[str] = ["LogisticRegression", "RandomForest"]):
        """
        Run progressive validation across multiple years.

        Args:
            all_data: All available data by year
            start_year: First year to validate (default: earliest year with enough history)
            end_year: Last year to validate (default: latest year)
            model_types: List of model types to train

        Returns:
            Dictionary with pipeline results
        """
        # Get available years
        available_years = sorted(list(all_data.keys()), key=int)

        if start_year is None:
            # Skip first year (needs at least 1 year of history)
            start_year = available_years[1] if len(available_years) > 1 else available_years[0]

        if end_year is None:
            end_year = available_years[-1]

        start_idx = available_years.index(start_year)
        end_idx = available_years.index(end_year)

        validation_years = available_years[start_idx:end_idx + 1]

        print("\n" + "="*70)
        print("PROGRESSIVE VALIDATION PIPELINE")
        print("="*70)
        print(f"Years: {available_years}")
        print(f"Validation years: {validation_years}")
        print("="*70)

        pipeline_results = {}

        for year in validation_years:
            print(f"\n\nProcessing year: {year}")

            result = self.retrain_with_drift_awareness(
                all_data=all_data,
                current_year=year,
                model_types=model_types,
                force_retrain=False  # Use drift/performance-based decision
            )

            pipeline_results[year] = result

            # Pause for readability
            print("\n" + "-"*70)

        # Generate pipeline summary
        print("\n" + "="*70)
        print("PIPELINE SUMMARY")
        print("="*70)

        summary_data = []
        for year, result in pipeline_results.items():
            if result["status"] == "success":
                summary_data.append({
                    "year": year,
                    "status": "retrained",
                    "reason": result["retraining_event"]["retraining_reason"],
                    "drift_rate": result["retraining_event"].get("drift_rate", 0),
                    "training_years": len(result["retraining_event"]["training_years"])
                })
            elif result["status"] == "skipped":
                summary_data.append({
                    "year": year,
                    "status": "skipped",
                    "reason": result["reason"],
                    "drift_rate": result.get("drift_rate", 0),
                    "training_years": "N/A"
                })
            else:
                summary_data.append({
                    "year": year,
                    "status": result["status"],
                    "reason": result.get("reason", "unknown"),
                    "drift_rate": "N/A",
                    "training_years": "N/A"
                })

        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string())

        return {
            "pipeline_results": pipeline_results,
            "summary": summary_df,
            "model_registry": self.registry.list_models()
        }


def main():
    """Example usage of the drift-aware retraining system."""
    # This would typically be imported from your data loading module
    from data import get_all_years_data

    print("DRIFT-AWARE RETRAINING SYSTEM DEMONSTRATION")
    print("="*70)

    # Load data
    print("\nLoading data...")
    try:
        all_data = get_all_years_data("input_data")
        print(f"Loaded data for years: {list(all_data.keys())}")
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Using synthetic data for demonstration...")
        # Create synthetic data for demonstration
        all_data = {}
        for year in ["2014", "2015", "2016", "2017", "2018", "2019", "2020"]:
            n_samples = np.random.randint(100, 500)
            n_features = 50
            X = pd.DataFrame(np.random.randn(n_samples, n_features),
                           columns=[f"feature_{i}" for i in range(n_features)])
            y = pd.Series(np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1]))
            all_data[year] = (X, y)
        print(f"Created synthetic data for years: {list(all_data.keys())}")

    # Initialize retraining system
    retraining_system = DriftAwareRetrainingSystem(
        drift_threshold=0.3,  # 30% drift triggers retraining
        performance_degradation_threshold=0.05,  # 5% performance drop triggers retraining
        registry_path="model_registry_demo"
    )

    # Run progressive validation pipeline
    results = retraining_system.run_progressive_validation_pipeline(
        all_data=all_data,
        start_year="2016",  # Skip first few years for sufficient training history
        end_year="2020",
        model_types=["LogisticRegression", "RandomForest"]
    )

    # Show model registry
    print("\n" + "="*70)
    print("MODEL REGISTRY")
    print("="*70)
    registry_df = retraining_system.registry.list_models()
    if not registry_df.empty:
        print(registry_df[['model_name', 'version', 'validation_year',
                          'retraining_reason', 'is_active']].to_string())

        # Show performance history
        print("\n\nPerformance History:")
        for model_name in ["LogisticRegression", "RandomForest"]:
            history_df = retraining_system.registry.get_model_performance_history(model_name)
            if not history_df.empty:
                print(f"\n{model_name}:")
                print(history_df[['version', 'validation_year', 'accuracy',
                                 'f1_score', 'retraining_reason']].to_string())
    else:
        print("No models registered yet")

    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)
    print("\nThe drift-aware retraining system has been demonstrated.")
    print("Key features implemented:")
    print("1. Drift detection-based retraining triggers")
    print("2. Performance degradation monitoring")
    print("3. Progressive validation strategy")
    print("4. Model registry with versioning")
    print("5. Comprehensive logging and decision tracking")


if __name__ == "__main__":
    main()