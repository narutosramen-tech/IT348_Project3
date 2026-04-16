from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
from sklearn.pipeline import Pipeline
from preprocessor import FraudPreprocessor
from enhanced_pipeline import create_enhanced_log_reg_pipeline, create_enhanced_rf_pipeline, create_enhanced_gb_pipeline, BASIC_FEATURE_CONFIG
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, Any, Union, List
from dataset import Dataset
from matplotlib import pyplot as plt

# Optional imports for imbalanced-learn
try:
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    print("Warning: imblearn not available. Install with: pip install imbalanced-learn")
    SMOTE = None
    RandomOverSampler = None
    RandomUnderSampler = None
    ImbPipeline = None

# Constants
TIE_THRESHOLD = 0.015  # 1.5% difference considered a tie

# Optional imports for visualization (only needed if plotting confusion matrices)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    plt = None
    sns = None


def apply_sampling(X: pd.DataFrame, y: pd.Series,
                  sampling_method: str = "none",
                  sampling_strategy: str = 'auto',
                  random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply sampling techniques to handle class imbalance.

    Args:
        X: Features DataFrame
        y: Target Series
        sampling_method: One of "none", "oversample", "undersample", "smote"
        sampling_strategy: Target sampling ratio (minority/majority).
                          'auto' balances classes, 'majority' keeps majority class same,
                          or specify ratio like '0.5' or '0.25'.
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (sampled_X, sampled_y)
    """
    if sampling_method == "none":
        return X, y

    if not IMBLEARN_AVAILABLE:
        print("Warning: imblearn not available. Install with: pip install imbalanced-learn")
        return X, y

    # Handle None sampling strategy
    if sampling_strategy is None:
        sampling_strategy = 'auto'

    try:
        if sampling_method == "oversample":
            sampler = RandomOverSampler(
                sampling_strategy=sampling_strategy,
                random_state=random_state
            )
        elif sampling_method == "undersample":
            sampler = RandomUnderSampler(
                sampling_strategy=sampling_strategy,
                random_state=random_state
            )
        elif sampling_method == "smote":
            # Calculate safe k_neighbors
            minority_count = sum(y == 1)
            k_neighbors = min(5, minority_count - 1) if minority_count > 1 else 1

            sampler = SMOTE(
                sampling_strategy=sampling_strategy,
                random_state=random_state,
                k_neighbors=k_neighbors
            )
        else:
            raise ValueError(f"Unknown sampling method: {sampling_method}. "
                             f"Choose from: 'none', 'oversample', 'undersample', 'smote'")

        columns = X.columns

        X_resampled, y_resampled = sampler.fit_resample(X, y)

        # Convert back to pandas for consistency
        X_resampled = pd.DataFrame(X_resampled, columns=columns, index=range(len(X_resampled)))
        y_resampled = pd.Series(y_resampled, index=range(len(y_resampled)))

        print(f"Applied {sampling_method}: {len(X)} -> {len(X_resampled)} samples "
              f"({sum(y_resampled == 1)}/{len(y_resampled)} fraud ratio)")

        return X_resampled, y_resampled

    except Exception as e:
        print(f"Warning: Sampling failed: {e}. Returning original data.")
        return X, y


def create_sampling_pipeline(base_pipeline: Pipeline,
                            sampling_method: str = "none",
                            sampling_strategy: str = 'auto',
                            random_state: int = 42) -> Pipeline:
    """
    Create a pipeline with sampling step for imbalanced data.

    Args:
        base_pipeline: Base sklearn pipeline
        sampling_method: One of "none", "oversample", "undersample", "smote"
        sampling_strategy: Target sampling ratio
        random_state: Random seed

    Returns:
        Pipeline with sampling step (if imblearn available)
    """
    if sampling_method == "none":
        return base_pipeline

    if not IMBLEARN_AVAILABLE:
        print("Warning: imblearn not available. Install with: pip install imbalanced-learn")
        return base_pipeline

    # Check if required classes are available
    if RandomOverSampler is None or RandomUnderSampler is None or SMOTE is None or ImbPipeline is None:
        print("Warning: imblearn classes not available, returning base pipeline")
        return base_pipeline

    try:
        # Extract steps from base pipeline
        steps = []

        # Add sampling step
        if sampling_method == "oversample":
            sampler = RandomOverSampler(
                sampling_strategy=sampling_strategy,
                random_state=random_state
            )
        elif sampling_method == "undersample":
            sampler = RandomUnderSampler(
                sampling_strategy=sampling_strategy,
                random_state=random_state
            )
        elif sampling_method == "smote":
            sampler = SMOTE(
                sampling_strategy=sampling_strategy,
                random_state=random_state,
                k_neighbors=5  # Safe default
            )
        else:
            raise ValueError(f"Unknown sampling method: {sampling_method}")

        steps.append(('sampling', sampler))

        # Add steps from base pipeline
        for step_name, step_transformer in base_pipeline.steps:
            steps.append((step_name, step_transformer))

        return ImbPipeline(steps)

    except Exception as e:
        print(f"Warning: Failed to create sampling pipeline: {e}. Returning base pipeline.")
        return base_pipeline


class ClassifierEvaluator:
    """
    A class to evaluate classifier performance with metrics in order of precedence:
    Recall > F1-score > Precision > Accuracy

    For malware detection, minimizing false negatives (missed malware) is critical.
    Models within 1.5% (0.015) difference are considered tied for that metric.
    """

    def __init__(self, classifier_name: str, y_true, y_pred, y_pred_proba=None):
        """
        Initialize evaluator with ground truth and predictions.

        Args:
            classifier_name: Name of the classifier being evaluated
            y_true: Ground truth labels (numpy array, pandas Series, or list)
            y_pred: Predicted labels (numpy array, pandas Series, or list)
            y_pred_proba: Predicted probabilities for positive class (numpy array, optional)
        """
        self.classifier_name = classifier_name

        # Convert to numpy arrays for consistent handling
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.y_pred_proba = np.array(y_pred_proba) if y_pred_proba is not None else None

    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate all evaluation metrics.

        Returns:
            Dictionary with accuracy, f1, precision, recall scores
        """
        metrics = {}

        # Calculate accuracy
        metrics['accuracy'] = accuracy_score(self.y_true, self.y_pred)

        # Calculate F1-score, precision, recall (macro-averaged for imbalanced datasets)
        metrics['f1_score'] = f1_score(self.y_true, self.y_pred, average='binary', pos_label = 1)
        metrics['precision'] = precision_score(self.y_true, self.y_pred, average='binary', pos_label = 1)
        metrics['recall'] = recall_score(self.y_true, self.y_pred, average='binary', pos_label = 1)

        # Calculate ROC-AUC and PR-AUC if probabilities are available
        if self.y_pred_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(self.y_true, self.y_pred_proba)
                metrics['pr_auc'] = average_precision_score(self.y_true, self.y_pred_proba)
            except Exception as e:
                print(f"Warning: Could not calculate AUC metrics: {e}")
                metrics['roc_auc'] = 0.0
                metrics['pr_auc'] = 0.0
        else:
            metrics['roc_auc'] = 0.0
            metrics['pr_auc'] = 0.0

        return metrics

    def get_confusion_matrix(self, normalize: bool = False) -> np.ndarray:
        """
        Generate confusion matrix.

        Args:
            normalize: If True, return normalized confusion matrix

        Returns:
            Confusion matrix as numpy array
        """
        cm = confusion_matrix(self.y_true, self.y_pred)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        return cm

    def plot_confusion_matrix(self, save_path: Optional[str] = None,
                              normalize: bool = False, title_suffix: str = "") -> None:
        """
        Plot confusion matrix as a heatmap.

        Args:
            save_path: Optional path to save the plot image
            normalize: If True, plot normalized confusion matrix
            title_suffix: Additional text to append to title
        """
        if not VISUALIZATION_AVAILABLE or plt is None:
            print("Visualization libraries (matplotlib, seaborn) not available.")
            print("Install with: pip install matplotlib seaborn")
            return

        cm = self.get_confusion_matrix(normalize=normalize)

        plt.figure(figsize=(8, 6))

        if normalize:
            display_cm = np.round(cm, 2)
            title = f"Normalized Confusion Matrix - {self.classifier_name} {title_suffix}"
        else:
            display_cm = cm
            title = f"Confusion Matrix - {self.classifier_name} {title_suffix}"

        sns.heatmap(display_cm, annot=True, fmt='.2f' if normalize else 'd',
                    cmap='Blues', linewidths=0.5, linecolor='gray')

        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")

        plt.show()

    def plot_roc_curve(self, save_path: Optional[str] = None) -> None:
        """
        Plot ROC curve for binary classification.

        Args:
            save_path: Optional path to save the plot image
        """
        if not VISUALIZATION_AVAILABLE or plt is None:
            print("Visualization libraries (matplotlib, seaborn) not available.")
            print("Install with: pip install matplotlib seaborn")
            return

        if self.y_pred_proba is None:
            print("Warning: No predicted probabilities available for ROC curve.")
            print("ROC curve requires predict_proba output from classifier.")
            return

        from sklearn.metrics import roc_curve

        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_pred_proba)

        plt.figure(figsize=(8, 6))

        # Plot ROC curve
        metrics = self.calculate_metrics()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {metrics.get("roc_auc", 0.0):.3f})')

        # Plot diagonal line (random classifier)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {self.classifier_name}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to {save_path}")

        plt.show()

    def plot_precision_recall_curve(self, save_path: Optional[str] = None) -> None:
        """
        Plot Precision-Recall curve for binary classification.

        Args:
            save_path: Optional path to save the plot image
        """
        if not VISUALIZATION_AVAILABLE or plt is None:
            print("Visualization libraries (matplotlib, seaborn) not available.")
            print("Install with: pip install matplotlib seaborn")
            return

        if self.y_pred_proba is None:
            print("Warning: No predicted probabilities available for Precision-Recall curve.")
            print("Precision-Recall curve requires predict_proba output from classifier.")
            return

        from sklearn.metrics import precision_recall_curve

        # Calculate Precision-Recall curve
        precision, recall, thresholds = precision_recall_curve(self.y_true, self.y_pred_proba)

        plt.figure(figsize=(8, 6))

        # Plot Precision-Recall curve
        metrics = self.calculate_metrics()
        plt.plot(recall, precision, color='darkgreen', lw=2,
                 label=f'PR curve (AP = {metrics.get("pr_auc", 0.0):.3f})')

        # Plot no-skill line (baseline)
        baseline_precision = self.y_true.sum() / len(self.y_true)
        plt.axhline(y=baseline_precision, color='navy', lw=2, linestyle='--',
                   label=f'Baseline (precision = {baseline_precision:.3f})')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {self.classifier_name}')
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Precision-Recall curve saved to {save_path}")

        plt.show()

    def plot_metrics_comparison(self, other_evaluator: Optional['ClassifierEvaluator'] = None,
                               save_path: Optional[str] = None) -> None:
        """
        Plot comparison of metrics between this classifier and optionally another.

        Args:
            other_evaluator: Optional other ClassifierEvaluator for comparison
            save_path: Optional path to save the plot image
        """
        if not VISUALIZATION_AVAILABLE or plt is None:
            print("Visualization libraries (matplotlib, seaborn) not available.")
            print("Install with: pip install matplotlib seaborn")
            return

        # Get metrics for this classifier
        metrics = self.calculate_metrics()

        # Prepare data for plotting
        classifier_names = [self.classifier_name]
        metric_data = {
            'accuracy': [metrics.get('accuracy', 0.0)],
            'precision': [metrics.get('precision', 0.0)],
            'recall': [metrics.get('recall', 0.0)],
            'f1_score': [metrics.get('f1_score', 0.0)]
        }

        if other_evaluator:
            other_metrics = other_evaluator.calculate_metrics()
            classifier_names.append(other_evaluator.classifier_name)
            metric_data['accuracy'].append(other_metrics.get('accuracy', 0.0))
            metric_data['precision'].append(other_metrics.get('precision', 0.0))
            metric_data['recall'].append(other_metrics.get('recall', 0.0))
            metric_data['f1_score'].append(other_metrics.get('f1_score', 0.0))

        # Create grouped bar chart
        x = np.arange(len(classifier_names))
        width = 0.2

        plt.figure(figsize=(10, 6))

        plt.bar(x - 1.5*width, metric_data['accuracy'], width, label='Accuracy', color='skyblue')
        plt.bar(x - 0.5*width, metric_data['precision'], width, label='Precision', color='lightcoral')
        plt.bar(x + 0.5*width, metric_data['recall'], width, label='Recall', color='lightgreen')
        plt.bar(x + 1.5*width, metric_data['f1_score'], width, label='F1-Score', color='gold')

        plt.xlabel('Classifier')
        plt.ylabel('Score')
        plt.title('Classifier Performance Comparison')
        plt.xticks(x, classifier_names)
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        plt.ylim([0, 1])
        plt.grid(True, alpha=0.3, axis='y')

        # Add value labels on top of bars
        for i in range(len(classifier_names)):
            for j, (metric_name, values) in enumerate(metric_data.items()):
                height = values[i]
                plt.text(x[i] + (j - 1.5)*width, height + 0.01, f'{height:.3f}',
                         ha='center', va='bottom', fontsize=8)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Metrics comparison plot saved to {save_path}")

        plt.show()

    def plot_threshold_analysis(self, save_path: Optional[str] = None) -> None:
        """
        Analyze how different probability thresholds affect precision, recall, F1.

        Args:
            save_path: Optional path to save the plot image
        """
        if not VISUALIZATION_AVAILABLE or plt is None:
            print("Visualization libraries (matplotlib, seaborn) not available.")
            print("Install with: pip install matplotlib seaborn")
            return

        if self.y_pred_proba is None:
            print("Warning: No predicted probabilities available for threshold analysis.")
            return

        from sklearn.metrics import precision_recall_curve, f1_score

        precision, recall, thresholds = precision_recall_curve(self.y_true, self.y_pred_proba)
        f1_scores = []

        # Calculate F1 scores for each threshold
        for i in range(len(thresholds)):
            y_pred_threshold = (self.y_pred_proba >= thresholds[i]).astype(int)
            f1_scores.append(f1_score(self.y_true, y_pred_threshold))

        thresholds = thresholds[:-1]  # Remove last threshold (undefined)

        plt.figure(figsize=(12, 8))

        # Plot precision, recall, and F1 vs threshold
        plt.plot(thresholds, precision[:-1], 'b-', label='Precision', lw=2)
        plt.plot(thresholds, recall[:-1], 'g-', label='Recall', lw=2)
        plt.plot(thresholds, f1_scores, 'r-', label='F1-Score', lw=2)

        # Find optimal threshold (max F1)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_f1 = f1_scores[optimal_idx]

        # Mark optimal point
        plt.axvline(x=optimal_threshold, color='k', linestyle='--', alpha=0.7)
        plt.scatter(optimal_threshold, optimal_f1, color='red', s=100, zorder=5)

        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title(f'Threshold Analysis - {self.classifier_name}')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)

        # Add text annotation for optimal threshold
        plt.annotate(f'Optimal threshold: {optimal_threshold:.3f}\nF1: {optimal_f1:.3f}',
                    xy=(optimal_threshold, optimal_f1),
                    xytext=(optimal_threshold + 0.05, optimal_f1 - 0.1),
                    arrowprops=dict(arrowstyle="->", color='red', lw=1),
                    fontsize=10)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Threshold analysis plot saved to {save_path}")

        plt.show()

    def visualize_all(self, save_dir: Optional[str] = None) -> None:
        """
        Generate all available visualizations for this classifier.

        Args:
            save_dir: Optional directory to save all plots
        """
        if not VISUALIZATION_AVAILABLE or plt is None:
            print("Visualization libraries (matplotlib, seaborn) not available.")
            print("Install with: pip install matplotlib seaborn")
            return

        # Create save paths if save_dir is provided
        save_paths = {}
        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            save_paths = {
                'confusion_matrix': os.path.join(save_dir, f'{self.classifier_name}_confusion_matrix.png'),
                'roc_curve': os.path.join(save_dir, f'{self.classifier_name}_roc_curve.png'),
                'pr_curve': os.path.join(save_dir, f'{self.classifier_name}_pr_curve.png'),
                'threshold_analysis': os.path.join(save_dir, f'{self.classifier_name}_threshold_analysis.png')
            }

        # Generate all visualizations
        print(f"\n{'='*60}")
        print(f"Generating all visualizations for {self.classifier_name}")
        print(f"{'='*60}")

        # Confusion Matrix
        print("\n1. Confusion Matrix:")
        self.plot_confusion_matrix(save_path=save_paths.get('confusion_matrix'))

        # ROC Curve (if probabilities available)
        if self.y_pred_proba is not None:
            print("\n2. ROC Curve:")
            self.plot_roc_curve(save_path=save_paths.get('roc_curve'))

        # Precision-Recall Curve (if probabilities available)
        if self.y_pred_proba is not None:
            print("\n3. Precision-Recall Curve:")
            self.plot_precision_recall_curve(save_path=save_paths.get('pr_curve'))

        # Threshold Analysis (if probabilities available)
        if self.y_pred_proba is not None:
            print("\n4. Threshold Analysis:")
            self.plot_threshold_analysis(save_path=save_paths.get('threshold_analysis'))

        print(f"\n{'='*60}")
        print("All visualizations completed!")
        print(f"{'='*60}")

    def plot_feature_importance(self, model: Any = None, feature_names: Optional[List[str]] = None,
                               top_n: int = 20, save_path: Optional[str] = None) -> None:
        """
        Plot feature importance if model supports it.

        Args:
            model: Trained model object (must have feature_importances_ or coef_ attribute)
            feature_names: Optional list of feature names
            top_n: Number of top features to display
            save_path: Optional path to save the plot image
        """
        if not VISUALIZATION_AVAILABLE or plt is None:
            print("Visualization libraries (matplotlib, seaborn) not available.")
            print("Install with: pip install matplotlib seaborn")
            return

        if model is None:
            print("Warning: No model provided for feature importance analysis.")
            return

        # Try to get feature importances
        importances = None

        # Check for different types of feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # For linear models, take absolute value of coefficients
            coef = model.coef_
            if len(coef.shape) > 1:
                # For multi-class, get average magnitude across classes
                importances = np.mean(np.abs(coef), axis=0)
            else:
                importances = np.abs(coef)

        if importances is None:
            print(f"Warning: Model {type(model).__name__} does not have feature_importances_ or coef_ attribute.")
            return

        # If no feature names provided, use generic names
        actual_feature_names: List[str]
        if feature_names is None:
            actual_feature_names = [f'Feature {i}' for i in range(len(importances))]
        else:
            actual_feature_names = feature_names

        # Sort features by importance
        indices = np.argsort(importances)[::-1]

        # Take top N features
        top_indices = indices[:top_n]
        top_importances = importances[top_indices]
        top_names = [actual_feature_names[i] for i in top_indices]

        # Create horizontal bar plot
        plt.figure(figsize=(10, max(6, top_n * 0.3)))

        y_pos = np.arange(len(top_importances))
        plt.barh(y_pos, top_importances, align='center', color='steelblue', alpha=0.8)
        plt.yticks(y_pos, top_names)
        plt.xlabel('Importance')
        plt.title(f'Feature Importance - {self.classifier_name}')
        plt.gca().invert_yaxis()  # Most important at top

        # Add value labels
        for i, v in enumerate(top_importances):
            plt.text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to {save_path}")

        plt.show()

    def evaluate(self, verbose: bool = True,
                 include_confusion_matrix: bool = False,
                 plot_confusion_matrix: bool = False,
                 plot_roc_curve: bool = False,
                 plot_pr_curve: bool = False,
                 plot_threshold_analysis: bool = False,
                 plot_metrics_comparison: Optional['ClassifierEvaluator'] = None) -> Dict[str, Any]:
        """
        Comprehensive evaluation of classifier performance.

        Args:
            verbose: If True, print detailed results
            include_confusion_matrix: If True, include confusion matrix in results
            plot_confusion_matrix: If True, plot confusion matrix
            plot_roc_curve: If True, plot ROC curve (requires predicted probabilities)
            plot_pr_curve: If True, plot Precision-Recall curve (requires predicted probabilities)
            plot_threshold_analysis: If True, plot threshold analysis (requires predicted probabilities)
            plot_metrics_comparison: Optional other ClassifierEvaluator for metrics comparison plot

        Returns:
            Dictionary with all evaluation metrics and optionally confusion matrix
        """
        # Calculate metrics
        metrics = self.calculate_metrics()

        if verbose:
            self._print_evaluation_report(metrics)

        # Get confusion matrix if requested
        results = {'metrics': metrics, 'classifier_name': self.classifier_name}

        if include_confusion_matrix or plot_confusion_matrix:
            cm = self.get_confusion_matrix()
            results['confusion_matrix'] = cm

        # Generate requested visualizations
        if plot_confusion_matrix:
            self.plot_confusion_matrix()

        if plot_roc_curve:
            if self.y_pred_proba is not None:
                self.plot_roc_curve()
            elif verbose:
                print("Skipping ROC curve: No predicted probabilities available.")

        if plot_pr_curve:
            if self.y_pred_proba is not None:
                self.plot_precision_recall_curve()
            elif verbose:
                print("Skipping Precision-Recall curve: No predicted probabilities available.")

        if plot_threshold_analysis:
            if self.y_pred_proba is not None:
                self.plot_threshold_analysis()
            elif verbose:
                print("Skipping threshold analysis: No predicted probabilities available.")

        if plot_metrics_comparison is not None:
            self.plot_metrics_comparison(other_evaluator=plot_metrics_comparison)

        return results

    def _print_evaluation_report(self, metrics: Dict[str, float]) -> None:
        """
        Print formatted evaluation report.
        """
        print(f"\n{'='*60}")
        print(f"EVALUATION RESULTS - {self.classifier_name}")
        print(f"{'='*60}")

        # Print metrics in new precedence order (Recall first for malware detection)
        print(f"Recall*:   {metrics['recall']:.4f}  (Primary: Minimizing missed malware)")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")

        # Print AUC metrics if available
        if metrics.get('roc_auc', 0) > 0:
            print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
            print(f"PR-AUC:    {metrics['pr_auc']:.4f}")
        else:
            print("AUC Metrics: Not available (probabilities required)")

        print(f"{'='*60}")

    def compare_with_other(self, other_evaluator: 'ClassifierEvaluator',
                           verbose: bool = True) -> Dict[str, str]:
        """
        Compare this classifier with another based on precedence order.

        Args:
            other_evaluator: Another ClassifierEvaluator instance to compare with
            verbose: If True, print comparison results

        Returns:
            Dictionary with comparison results and winner for each metric
        """
        metrics_self = self.calculate_metrics()
        metrics_other = other_evaluator.calculate_metrics()

        comparison = {}

        # Compare in NEW precedence order: Recall > F1 > Precision > Accuracy (malware detection focus)
        # Use tie threshold for all comparisons

        # Recall (most important for malware detection)
        recall_comparison = self._is_better_metric(metrics_self['recall'], metrics_other['recall'])
        comparison['recall_winner'] = self._get_winner_from_comparison(recall_comparison, other_evaluator.classifier_name)

        # F1-Score (balanced measure)
        f1_comparison = self._is_better_metric(metrics_self['f1_score'], metrics_other['f1_score'])
        comparison['f1_winner'] = self._get_winner_from_comparison(f1_comparison, other_evaluator.classifier_name)

        # Precision (minimize false positives)
        precision_comparison = self._is_better_metric(metrics_self['precision'], metrics_other['precision'])
        comparison['precision_winner'] = self._get_winner_from_comparison(precision_comparison, other_evaluator.classifier_name)

        # Accuracy (overall performance)
        accuracy_comparison = self._is_better_metric(metrics_self['accuracy'], metrics_other['accuracy'])
        comparison['accuracy_winner'] = self._get_winner_from_comparison(accuracy_comparison, other_evaluator.classifier_name)

        # AUC metrics comparison (if available)
        roc_auc_comparison = self._is_better_metric(metrics_self['roc_auc'], metrics_other['roc_auc'])
        comparison['roc_auc_winner'] = self._get_winner_from_comparison(roc_auc_comparison, other_evaluator.classifier_name)

        pr_auc_comparison = self._is_better_metric(metrics_self['pr_auc'], metrics_other['pr_auc'])
        comparison['pr_auc_winner'] = self._get_winner_from_comparison(pr_auc_comparison, other_evaluator.classifier_name)

        # Determine overall winner based on precedence
        overall_winner = self._determine_overall_winner(comparison)
        comparison['overall_winner'] = overall_winner

        if verbose:
            self._print_comparison_report(comparison, metrics_self, metrics_other,
                                         other_evaluator.classifier_name)

        return comparison

    def _is_better_metric(self, metric_self: float, metric_other: float) -> str:
        """
        Determine winner for a metric with tie threshold.

        Returns:
            'self', 'other', or 'tie' if within tie threshold
        """
        diff = abs(metric_self - metric_other)

        if diff <= TIE_THRESHOLD:
            return 'tie'
        elif metric_self > metric_other:
            return 'self'
        else:
            return 'other'

    def _get_winner_from_comparison(self, comparison_result: str, other_name: str) -> str:
        """Convert comparison result to winner string."""
        if comparison_result == 'self':
            return self.classifier_name
        elif comparison_result == 'other':
            return other_name
        else:  # 'tie'
            return "Tie"

    def _determine_overall_winner(self, comparison: Dict[str, str]) -> str:
        """
        Determine overall winner based on precedence order.
        For fraud detection: Recall > ROC-AUC > PR-AUC > F1 > Precision > Accuracy
        """
        # Check in order: Recall > ROC-AUC > PR-AUC > F1 > Precision > Accuracy (fraud detection focus)
        if comparison['recall_winner'] != "Tie":
            return comparison['recall_winner']
        elif comparison['roc_auc_winner'] != "Tie":
            return comparison['roc_auc_winner']
        elif comparison['pr_auc_winner'] != "Tie":
            return comparison['pr_auc_winner']
        elif comparison['f1_winner'] != "Tie":
            return comparison['f1_winner']
        elif comparison['precision_winner'] != "Tie":
            return comparison['precision_winner']
        elif comparison['accuracy_winner'] != "Tie":
            return comparison['accuracy_winner']
        else:
            return "Tie (All metrics within 1.5% threshold)"

    def _print_comparison_report(self, comparison: Dict[str, str],
                                metrics_self: Dict[str, float],
                                metrics_other: Dict[str, float],
                                other_name: str) -> None:
        """
        Print formatted comparison report.
        """
        print(f"\n{'='*60}")
        print(f"CLASSIFIER COMPARISON")
        print(f"{'='*60}")
        print(f"{self.classifier_name} vs {other_name}")
        print(f"{'='*60}")

        print(f"\nRecall* Comparison (Primary for malware detection):")
        print(f"  {self.classifier_name}: {metrics_self['recall']:.4f}")
        print(f"  {other_name}: {metrics_other['recall']:.4f}")
        print(f"  Winner: {comparison['recall_winner']}")
        print(f"     (Models within {TIE_THRESHOLD:.3f} difference considered tied)")

        print(f"\nF1-Score Comparison (Balanced measure):")
        print(f"  {self.classifier_name}: {metrics_self['f1_score']:.4f}")
        print(f"  {other_name}: {metrics_other['f1_score']:.4f}")
        print(f"  Winner: {comparison['f1_winner']}")

        print(f"\nPrecision Comparison (Minimizing false positives):")
        print(f"  {self.classifier_name}: {metrics_self['precision']:.4f}")
        print(f"  {other_name}: {metrics_other['precision']:.4f}")
        print(f"  Winner: {comparison['precision_winner']}")

        print(f"\nAccuracy Comparison (Overall performance):")
        print(f"  {self.classifier_name}: {metrics_self['accuracy']:.4f}")
        print(f"  {other_name}: {metrics_other['accuracy']:.4f}")
        print(f"  Winner: {comparison['accuracy_winner']}")

        # AUC metrics comparison (if available)
        if metrics_self.get('roc_auc', 0) > 0 and metrics_other.get('roc_auc', 0) > 0:
            print(f"\nROC-AUC Comparison (Overall discriminative ability):")
            print(f"  {self.classifier_name}: {metrics_self['roc_auc']:.4f}")
            print(f"  {other_name}: {metrics_other['roc_auc']:.4f}")
            print(f"  Winner: {comparison['roc_auc_winner']}")

            print(f"\nPR-AUC Comparison (Precision-Recall tradeoff):")
            print(f"  {self.classifier_name}: {metrics_self['pr_auc']:.4f}")
            print(f"  {other_name}: {metrics_other['pr_auc']:.4f}")
            print(f"  Winner: {comparison['pr_auc_winner']}")
        else:
            print(f"\nAUC Metrics: Not available for one or both models (probabilities required)")

        print(f"\n{'='*60}")
        print(f"OVERALL WINNER (Recall > ROC-AUC > PR-AUC > F1 > Precision > Accuracy): {comparison['overall_winner']}")
        print(f"Tie threshold: {TIE_THRESHOLD:.3f} (1.5%)")
        print(f"{'='*60}")


def train_and_evaluate_classifiers_with_sampling(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    sampling_method: str = "none",
    sampling_strategy: str = 'auto',
    verbose: bool = True,
    random_state: int = 42
) -> Dict[str, Dict[str, Any]]:
    """
    Train multiple classifiers with optional sampling for class imbalance.

    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        sampling_method: One of "none", "oversample", "undersample", "smote"
        sampling_strategy: Sampling strategy (default: 'auto')
        verbose: If True, print detailed results
        random_state: Random seed

    Returns:
        Dictionary with trained models and evaluation results
    """
    print(f"\nTraining classifiers with {sampling_method} sampling...")

    # Apply sampling to training data
    if sampling_method != "none":
        X_train_resampled, y_train_resampled = apply_sampling(
            X_train, y_train,
            sampling_method=sampling_method,
            sampling_strategy=sampling_strategy,
            random_state=random_state
        )
    else:
        X_train_resampled, y_train_resampled = X_train, y_train

    # Create enhanced pipelines with optional sampling step
    pipelines = {
        "LogReg": create_sampling_pipeline(
            create_enhanced_log_reg_pipeline(BASIC_FEATURE_CONFIG),
            sampling_method=sampling_method,
            sampling_strategy=sampling_strategy,
            random_state=random_state
        ),

        "RandomForest": create_sampling_pipeline(
            create_enhanced_rf_pipeline(BASIC_FEATURE_CONFIG),
            sampling_method=sampling_method,
            sampling_strategy=sampling_strategy,
            random_state=random_state
        ),

        "GradientBoosting": create_sampling_pipeline(
            create_enhanced_gb_pipeline(BASIC_FEATURE_CONFIG),
            sampling_method=sampling_method,
            sampling_strategy=sampling_strategy,
            random_state=random_state
        )
    }

    results = {}

    for name, pipe in pipelines.items():
        # For imblearn pipelines with sampling, we need to fit with resampled data
        if sampling_method != "none" and IMBLEARN_AVAILABLE and ImbPipeline is not None:
            pipe.fit(X_train_resampled, y_train_resampled)
        else:
            pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)

        # Get probabilities if available for AUC metrics
        y_pred_proba = None
        try:
            if hasattr(pipe, 'predict_proba'):
                y_pred_proba = pipe.predict_proba(X_test)[:, 1]
        except:
            y_pred_proba = None

        evaluator = ClassifierEvaluator(name, y_test, y_pred, y_pred_proba)
        eval_result = evaluator.evaluate(verbose=verbose)

        results[name] = {
            "model": pipe,
            "metrics": eval_result["metrics"],
            "evaluator": evaluator
        }

    return results


def train_and_evaluate_classifiers(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    verbose: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Train multiple classifiers and evaluate performance using a consistent structure.

    Returns:
        {
            model_name: {
                "model": trained_model,
                "predictions": np.ndarray,
                "metrics": {recall, f1_score, precision, accuracy},
                "evaluator": ClassifierEvaluator,
                "confusion_matrix": np.ndarray
            }
        }
    """

    pipelines = {
        "LogReg": create_enhanced_log_reg_pipeline(BASIC_FEATURE_CONFIG),
        "RandomForest": create_enhanced_rf_pipeline(BASIC_FEATURE_CONFIG),
        "GradientBoosting": create_enhanced_gb_pipeline(BASIC_FEATURE_CONFIG)
    }

    results = {}

    for name, pipe in pipelines.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        # Get probabilities if available for AUC metrics
        y_pred_proba = None
        try:
            if hasattr(pipe, 'predict_proba'):
                y_pred_proba = pipe.predict_proba(X_test)[:, 1]
        except:
            y_pred_proba = None

        evaluator = ClassifierEvaluator(name, y_test, y_pred, y_pred_proba)
        eval_result = evaluator.evaluate(verbose=True)

        results[name] = {
            "model": pipe,
            "metrics": eval_result["metrics"],
            "evaluator": evaluator
        }

    return results

def train_from_dataset_with_sampling(
    dataset: Dataset,
    sample_name: str,
    sampling_method: str = "none",
    sampling_strategy: str = 'auto',
    random_state: int = 42
):
    """
    Train classifiers from dataset with optional sampling.
    """
    X_train, X_test, y_train, y_test = dataset.train_test_split(sample_name)

    return train_and_evaluate_classifiers_with_sampling(
        X_train, X_test, y_train, y_test,
        sampling_method=sampling_method,
        sampling_strategy=sampling_strategy,
        random_state=random_state
    )


def train_from_dataset(dataset: Dataset, sample_name: str):
    X_train, X_test, y_train, y_test = dataset.train_test_split(sample_name)

    return train_and_evaluate_classifiers(
        X_train, X_test, y_train, y_test
    )


def quick_evaluate_classifier(classifier_name: str, y_true: np.ndarray, y_pred: np.ndarray,
                             y_pred_proba: Optional[np.ndarray] = None,
                             plot_cm: bool = False, normalize_cm: bool = False,
                             plot_roc_curve: bool = False,
                             plot_pr_curve: bool = False,
                             plot_threshold_analysis: bool = False) -> Dict[str, Any]:
    """
    Quick evaluation of a single classifier using the ClassifierEvaluator.

    Args:
        classifier_name: Name of the classifier
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities for positive class (optional)
        plot_cm: If True, plot confusion matrix
        normalize_cm: If True, normalize confusion matrix
        plot_roc_curve: If True, plot ROC curve (requires predicted probabilities)
        plot_pr_curve: If True, plot Precision-Recall curve (requires predicted probabilities)
        plot_threshold_analysis: If True, plot threshold analysis (requires predicted probabilities)

    Returns:
        Evaluation results
    """
    evaluator = ClassifierEvaluator(classifier_name, y_true, y_pred, y_pred_proba)

    if plot_cm:
        evaluator.plot_confusion_matrix(normalize=normalize_cm)

    return evaluator.evaluate(verbose=True, include_confusion_matrix=True,
                             plot_confusion_matrix=False,  # Already handled above
                             plot_roc_curve=plot_roc_curve,
                             plot_pr_curve=plot_pr_curve,
                             plot_threshold_analysis=plot_threshold_analysis)


class SecurityFirstEnsemble:
    """
    Security-first voting ensemble for fraud detection.
    Uses 3 models with security-conservative tie-breaking.
    """

    def __init__(self, tie_breaker: str = "malware",  # "malware", "reject", or "confidence"
                 voting_type: str = "hard",  # "hard", "soft", or "stacked"
                 sampling_method: str = "none",  # "none", "oversample", "undersample", "smote"
                 sampling_strategy: str = 'auto',  # Sampling strategy
                 random_state: int = 42):  # Random seed
        """
        Initialize ensemble.

        Args:
            tie_breaker: What to do when models disagree
                - "malware": Default to malware (security-first)
                - "reject": Flag for human review
                - "confidence": Use highest confidence prediction
            voting_type: Type of ensemble
                - "hard": Majority voting
                - "soft": Probability-weighted voting
                - "stacked": Stacked ensemble with meta-classifier
            sampling_method: Sampling technique for handling class imbalance
            sampling_strategy: Sampling strategy
            random_state: Random seed for reproducibility
        """
        self.tie_breaker = tie_breaker
        self.voting_type = voting_type
        self.sampling_method = sampling_method
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.models = self._create_base_models()
        self.ensemble = None
        self.is_fitted = False
        self.threshold = 0.45

    def _create_base_models(self) -> Dict[str, Any]:
        """Create the 3-model ensemble using enhanced pipelines."""
        models = {
            "LogisticRegression": create_enhanced_log_reg_pipeline(BASIC_FEATURE_CONFIG),
            "RandomForest": create_enhanced_rf_pipeline(BASIC_FEATURE_CONFIG),
            "GradientBoosting": create_enhanced_gb_pipeline(BASIC_FEATURE_CONFIG)
        }
        return models

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Train all models in the ensemble."""
        print(f"\n{'='*70}")
        print(f"TRAINING 3-MODEL ENSEMBLE FOR MALWARE DETECTION")
        print(f"Sampling: {self.sampling_method} | Voting: {self.voting_type}")
        print(f"{'='*70}")

        # Apply sampling if requested
        if self.sampling_method != "none":
            print(f"\nApplying {self.sampling_method} sampling...")
            X_resampled, y_resampled = apply_sampling(
                X, y,
                sampling_method=self.sampling_method,
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state
            )
        else:
            X_resampled, y_resampled = X, y

        # Train individual models
        self.individual_predictions = {}
        self.individual_models = {}

        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            model.fit(X_resampled, y_resampled)
            self.individual_models[name] = model

            # Make predictions for ensemble voting
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X)[:, 1]  # Probability of malware class
                self.individual_predictions[name] = {
                    'predictions': model.predict(X),
                    'probabilities': probs
                }
            else:
                self.individual_predictions[name] = {
                    'predictions': model.predict(X),
                    'probabilities': None
                }

            # Evaluate individual performance
            y_pred = model.predict(X)
            y_pred_proba = None
            if hasattr(model, 'predict_proba'):
                try:
                    y_pred_proba = model.predict_proba(X)[:, 1]
                except:
                    y_pred_proba = None

            evaluator = ClassifierEvaluator(name, y, y_pred, y_pred_proba)
            results = evaluator.evaluate(verbose=False)
            print(f"  Accuracy: {results['metrics']['accuracy']:.4f}")
            print(f"  F1-Score: {results['metrics']['f1_score']:.4f}")

        # Create and train the ensemble based on voting type
        if self.voting_type == "hard":
            self._create_hard_voting_ensemble(X, y)
        elif self.voting_type == "soft":
            self._create_soft_voting_ensemble(X, y)
        elif self.voting_type == "stacked":
            self._create_stacked_ensemble(X, y)

        self.is_fitted = True
        print(f"\nEnsemble training complete. Tie-breaking: {self.tie_breaker}")

    def _create_hard_voting_ensemble(self, X: pd.DataFrame, y: pd.Series):
        """Create hard voting ensemble (majority vote)."""
        estimators = [(name, model) for name, model in self.individual_models.items()]
        self.ensemble = VotingClassifier(
            estimators=estimators,
            voting='hard',
            n_jobs=-1
        )
        self.ensemble.fit(X, y)

    def _create_soft_voting_ensemble(self, X: pd.DataFrame, y: pd.Series):
        """Create soft voting ensemble (probability weighted)."""
        estimators = [(name, model) for name, model in self.individual_models.items()]
        self.ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft',
            n_jobs=-1
        )
        self.ensemble.fit(X, y)

    def _create_stacked_ensemble(self, X: pd.DataFrame, y: pd.Series):
        """Create stacked ensemble with meta-classifier."""
        estimators = [(name, model) for name, model in self.individual_models.items()]
        self.ensemble = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=1000, random_state=42),
            cv=5,
            n_jobs=-1
        )
        self.ensemble.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with security-first tie-breaking."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before predicting")

        if self.ensemble is not None and self.voting_type in ["hard", "soft", "stacked"]:
            # Use sklearn's voting classifier if available
            predictions = self.ensemble.predict(X)
        else:
            # Custom voting with security-first tie-breaking
            predictions = self._security_first_vote(X)

        return np.array(predictions)
    
    def predict_threshold(
            self, 
            X: pd.DataFrame, 
            threshold: Optional[float] = None) -> np.ndarray:
        """
        Fraud decision layer based on probability threshold.
        Only applies to models that support probabilities.
        """
        if threshold is None:
            threshold = getattr(self, "threshold", 0.5)

        proba = self.predict_proba(X)[:, 1]

        return (proba >= threshold).astype(int)
        

    def _security_first_vote(self, X: pd.DataFrame) -> np.ndarray:
        """
        Custom voting with security-first tie-breaking.
        When models disagree, default to malware prediction.
        """
        # Collect predictions from all models
        all_predictions = []
        all_probabilities = []

        for name, model in self.individual_models.items():
            preds = model.predict(X)
            all_predictions.append(preds)

            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X)[:, 1]
                all_probabilities.append(probs)

        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)  # Shape: (n_models, n_samples)

        # Initialize final predictions
        final_predictions = np.zeros(X.shape[0], dtype=int)

        for i in range(X.shape[0]):
            model_votes = all_predictions[:, i]

            # Count votes
            malware_votes = np.sum(model_votes == 1)
            benign_votes = np.sum(model_votes == 0)

            if malware_votes > benign_votes:
                # Majority says malware
                final_predictions[i] = 1
            elif benign_votes > malware_votes:
                # Majority says benign
                final_predictions[i] = 0
            else:
                # Tie - use tie-breaking strategy
                final_predictions[i] = self._break_tie(i, all_predictions, all_probabilities)

        return final_predictions

    def _break_tie(self, sample_idx: int, all_predictions: np.ndarray,
                  all_probabilities: list) -> int:
        """Implement tie-breaking strategies."""
        if self.tie_breaker == "malware":
            # Security-first: default to malware
            return 1
        elif self.tie_breaker == "reject":
            # Flag for review (treat as malware for now)
            return 1
        elif self.tie_breaker == "confidence" and all_probabilities:
            # Use highest confidence
            confidences = []
            for probs in all_probabilities:
                if probs is not None:
                    # Convert to confidence: if prediction is 1, use prob; if 0, use 1-prob
                    pred = all_predictions[len(confidences), sample_idx]
                    conf = probs[sample_idx] if pred == 1 else 1 - probs[sample_idx]
                    confidences.append(conf)

            if confidences:
                # Use prediction from most confident model
                most_confident_idx = np.argmax(confidences)
                return all_predictions[most_confident_idx, sample_idx]
            else:
                # No probabilities available, default to malware
                return 1
        else:
            # Default to security-first
            return 1

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before predicting")

        if self.ensemble is not None and hasattr(self.ensemble, 'predict_proba'):
            return self.ensemble.predict_proba(X)
        else:
            # If ensemble doesn't have predict_proba, use average of base models
            all_probs = []
            for model in self.individual_models.values():
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(X)
                    all_probs.append(probs)

            if all_probs:
                return np.mean(all_probs, axis=0)
            else:
                # Fallback: create dummy probabilities
                preds = self.predict(X)
                probs = np.zeros((len(preds), 2))
                probs[preds == 0, 0] = 1.0
                probs[preds == 1, 1] = 1.0
                return probs

    def evaluate(self, X_test: pd.DataFrame, 
                 y_test: pd.Series,
                 verbose: bool = True,
                 threshold = 0.5
                 ) -> Dict[str, Any]:
        """Comprehensive evaluation of the ensemble."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before evaluating")

        print(f"\n{'='*70}")
        print("ENSEMBLE EVALUATION")
        print(f"{'='*70}")

        # Get ensemble predictions
        if self.voting_type == "soft":
            if threshold is not None:
                y_pred = self.predict_threshold(X_test, threshold=threshold)
            elif self.threshold is not None:
                y_pred = self.predict_threshold(X_test, threshold=self.threshold)
            else:
                y_pred = self.predict(X_test)
        else:
            y_pred = self.predict(X_test)

        # Use ClassifierEvaluator
        # Get probabilities if available for AUC metrics
        y_pred_proba = None
        if hasattr(self, 'predict_proba'):
            try:
                y_pred_proba = self.predict_proba(X_test)[:, 1]
            except:
                y_pred_proba = None

        evaluator = ClassifierEvaluator("SecurityFirstEnsemble", y_test, y_pred, y_pred_proba)
        results = evaluator.evaluate(verbose=verbose, include_confusion_matrix=True)

        # Add ensemble-specific metrics
        results['ensemble_type'] = self.voting_type
        results['tie_breaker'] = self.tie_breaker
        results['num_models'] = len(self.models)

        # Calculate model agreement
        if hasattr(self, 'individual_predictions'):
            agreement_rate = self._calculate_model_agreement(X_test, y_pred)
            results['model_agreement_rate'] = agreement_rate
            print(f"\nModel Agreement: {agreement_rate:.1%} of samples")

        # Show tie-breaking statistics
        if self.tie_breaker == "malware":
            print(f"Tie-breaking: Default to MALWARE (security-first)")

        return results

    def _calculate_model_agreement(self, X: pd.DataFrame, ensemble_preds: np.ndarray) -> float:
        """Calculate how often all models agree with each other."""
        if not hasattr(self, 'individual_models'):
            return 0.0

        all_preds = []
        for model in self.individual_models.values():
            all_preds.append(model.predict(X))
        
        # Stack into a matrix: (n_models, n_samples)
        pred_matrix = np.vstack(all_preds)

        # Check if all values in a column are equal to the first value in that column
        agreements = np.all(pred_matrix == pred_matrix[0, :], axis = 0)
        return np.mean(agreements)




def train_and_evaluate_ensemble_with_sampling(X_train: pd.DataFrame, X_test: pd.DataFrame,
                                            y_train: pd.Series, y_test: pd.Series,
                                            voting_type: str = "hard",
                                            tie_breaker: str = "malware",
                                            sampling_method: str = "none",
                                            sampling_strategy: str = 'auto',
                                            random_state: int = 42) -> Dict[str, Any]:
    """
    Train and evaluate the 3-model ensemble with optional sampling.

    Args:
        X_train, X_test: Training and test features
        y_train, y_test: Training and test labels
        voting_type: "hard", "soft", or "stacked"
        tie_breaker: "malware", "reject", or "confidence"
        sampling_method: One of "none", "oversample", "undersample", "smote"
        sampling_strategy: Sampling strategy
        random_state: Random seed

    Returns:
        Dictionary with ensemble results
    """
    print(f"\n{'='*70}")
    print(f"3-MODEL ENSEMBLE WITH {voting_type.upper()} VOTING")
    print(f"Sampling: {sampling_method} | Tie-breaker: {tie_breaker}")
    print(f"{'='*70}")

    # Create and train ensemble
    ensemble = SecurityFirstEnsemble(
        tie_breaker=tie_breaker,
        voting_type=voting_type,
        sampling_method=sampling_method,
        sampling_strategy=sampling_strategy,
        random_state=random_state
    )

    ensemble.fit(X_train, y_train)

    # Evaluate
    results = ensemble.evaluate(X_test, y_test, verbose=True)

    # Compare with individual models (with same sampling)
    print(f"\n{'='*70}")
    print("COMPARISON WITH INDIVIDUAL MODELS")
    print(f"{'='*70}")

    individual_results = train_and_evaluate_classifiers_with_sampling(
        X_train, X_test, y_train, y_test,
        sampling_method=sampling_method,
        sampling_strategy=sampling_strategy,
        random_state=random_state
    )

    # Find the best individual model using recall > f1 > precision > accuracy precedence
    best_score = (-1, -1, -1, -1)  # (Recall, F1, Precision, Accuracy)
    best_individual_model = None

    for model_name, model_results in individual_results.items():
        # Access metrics directly from the result dict
        metrics = model_results.get('metrics', {})
        if metrics:
            current_score = (
                metrics.get('recall', 0),
                metrics.get('f1_score', 0),
                metrics.get('precision', 0),
                metrics.get('accuracy', 0)
            )
            if current_score > best_score:
                best_score = current_score
                best_individual_model = model_name

    ensemble_metrics = results['metrics']
    ensemble_score = (ensemble_metrics['recall'],
                      ensemble_metrics['f1_score'],
                      ensemble_metrics['precision'],
                      ensemble_metrics['accuracy']
                )

    print(f"\nBest Individual Model ({best_individual_model}): Recall = {best_score[0]:.4f}, F1 = {best_score[1]:.4f}")
    print(f"Ensemble Model: Recall = {ensemble_score[0]:.4f}, F1 = {ensemble_score[1]:.4f}")

    if ensemble_score > best_score:
        print("Ensemble IS BETTER than the individual model based on prioritized metrics.")
    else:
        print("Ensemble IS NOT better than the best individual model.")
    return {
        'ensemble': ensemble,
        'ensemble_results': results,
        'individual_results': individual_results,
        'improvement': (ensemble_score[0] - best_score[0], ensemble_score[1] - best_score[1], ensemble_score[2] - best_score[2], ensemble_score[3] - best_score[3])
    }


def train_and_evaluate_ensemble(X_train: pd.DataFrame, X_test: pd.DataFrame,
                               y_train: pd.Series, y_test: pd.Series,
                               voting_type: str = "hard",
                               tie_breaker: str = "malware") -> Dict[str, Any]:
    """
    Train and evaluate the 3-model ensemble.

    Args:
        X_train, X_test: Training and test features
        y_train, y_test: Training and test labels
        voting_type: "hard", "soft", or "stacked"
        tie_breaker: "malware", "reject", or "confidence"

    Returns:
        Dictionary with ensemble results
    """
    print(f"\n{'='*70}")
    print(f"3-MODEL ENSEMBLE WITH {voting_type.upper()} VOTING")
    print(f"{'='*70}")

    # Create and train ensemble
    ensemble = SecurityFirstEnsemble(
        tie_breaker=tie_breaker,
        voting_type=voting_type
    )

    ensemble.fit(X_train, y_train)

    # Evaluate
    results = ensemble.evaluate(X_test, y_test, verbose=True)

    # Compare with individual models
    print(f"\n{'='*70}")
    print("COMPARISON WITH INDIVIDUAL MODELS")
    print(f"{'='*70}")

    individual_results = train_and_evaluate_classifiers(
        X_train, X_test, y_train, y_test,
    )

    # # Extract best F1 from individual models
    # best_individual_f1 = 0
    # Find the best individual model using recall > f1 > precision > accuracy precedence
    best_score = (-1, -1, -1, -1)  # (Recall, F1, Precision, Accuracy)
    best_individual_model = None

    for model_name, model_results in individual_results.items():
        # Access metrics directly from the result dict (pre-computed by train_and_evaluate_classifiers)
        metrics = model_results.get('metrics', {})
        if metrics:
            current_score = (
                metrics.get('recall', 0),
                metrics.get('f1_score', 0),
                metrics.get('precision', 0),
                metrics.get('accuracy', 0)
            )
            if current_score > best_score:
                best_score = current_score
                best_individual_model = model_name

    ensemble_metrics = results['metrics']
    ensemble_score = (ensemble_metrics['recall'], 
                      ensemble_metrics['f1_score'],
                      ensemble_metrics['precision'],
                      ensemble_metrics['accuracy']
                )

    print(f"\nBest Individual Model ({best_individual_model}): Recall = {best_score[0]:.4f}, F1 = {best_score[1]:.4f}")
    print(f"Ensemble Model: Recall = {ensemble_score[0]:.4f}, F1 = {ensemble_score[1]:.4f}")

    if ensemble_score > best_score:
        print("Ensemble IS BETTER than the individual model based on prioiritized metrics.")
    else:
        print("Ensemble IS NOT better than the best individual model.")
    return {
        'ensemble': ensemble,
        'ensemble_results': results,
        'individual_results': individual_results,
        'improvement': (ensemble_score[0] - best_score[0], ensemble_score[1] - best_score[1], ensemble_score[2] - best_score[2], ensemble_score[3] - best_score[3])
    }

def train_ensemble_from_dataset_with_sampling(
        dataset: Dataset, sample_name: str,
        voting_type: str = "hard",
        tie_breaker: str = "malware",
        sampling_method: str = "none",
        sampling_strategy: str = 'auto',
        random_state: int = 42
):
    """
    Train ensemble from dataset with optional sampling.
    """
    X_train, X_test, y_train, y_test = dataset.train_test_split(sample_name)

    return train_and_evaluate_ensemble_with_sampling(
        X_train, X_test, y_train, y_test,
        voting_type=voting_type,
        tie_breaker=tie_breaker,
        sampling_method=sampling_method,
        sampling_strategy=sampling_strategy,
        random_state=random_state
    )


def train_ensemble_from_dataset(
        dataset: Dataset, sample_name: str,
        voting_type: str = "hard",
        tie_breaker: str = "malware"
):
    X_train, X_test, y_train, y_test = dataset.train_test_split(sample_name)

    return train_and_evaluate_ensemble(
        X_train, X_test, y_train, y_test,
        voting_type=voting_type,
        tie_breaker=tie_breaker
    )