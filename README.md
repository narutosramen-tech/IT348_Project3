# Fraud Detection Machine Learning Project

A comprehensive fraud detection system with machine learning models, evaluation tools, and interactive CLI.

## Features

### Core Models
- **SecurityFirstEnsemble**: Voting ensemble for fraud detection with security-conservative tie-breaking
- **Individual Classifiers**: Logistic Regression, Random Forest, Gradient Boosting
- **Enhanced Pipelines**: Feature engineering and preprocessing pipelines

### Evaluation & Visualization
- **ClassifierEvaluator**: Comprehensive model evaluation with multiple metrics
- **Advanced Visualizations**: ROC curves, Precision-Recall curves, confusion matrices, feature importance plots
- **Metrics Comparison**: Compare performance between multiple models visually
- **Threshold Analysis**: Analyze how different probability thresholds affect metrics

### CLI Interface
- **Command-line Tools**: Train, test, compare, and evaluate models
- **Interactive Mode**: User-friendly interactive CLI
- **Visualization Commands**: Generate plots and save them to files
- **Batch Processing**: Process multiple models and datasets

## Installation

### Requirements
```bash
pip install -r requirements.txt
```

### Required Packages
```bash
pip install scikit-learn pandas numpy matplotlib seaborn imbalanced-learn
```

## Quick Start

### 1. Train a model
```bash
python fraud_cli.py train --transactions data/transactions.csv --identity data/identity.csv --voting hard
```

### 2. Test the model
```bash
python fraud_cli.py test --model saved_model.pkl --transactions test_transactions.csv --identity test_identity.csv
```

### 3. Generate visualizations
```bash
python fraud_cli.py visualize --model saved_model.pkl --transactions test_transactions.csv --identity test_identity.csv --save-plots-dir plots/
```

## CLI Commands

### train
Train a new ensemble model.

```bash
python fraud_cli.py train --transactions <path> --identity <path> [options]
```

**Options:**
- `--voting`: Voting type (hard/soft/stacked, default: hard)
- `--tie-breaker`: Tie-breaking strategy (malware/reject/confidence, default: malware)
- `--save-model`: Path to save trained model
- `--save-results`: Path to save training results
- `--plot-training-metrics`: Generate training visualizations
- `--save-plots-dir`: Directory to save plot images

### test
Test a trained model.

```bash
python fraud_cli.py test --model <path> --transactions <path> --identity <path> [options]
```

**Options:**
- `--threshold`: Probability threshold for classification (default: 0.5)
- `--plot-confusion-matrix`: Plot confusion matrix
- `--plot-roc-curve`: Plot ROC curve (requires predicted probabilities)
- `--plot-pr-curve`: Plot Precision-Recall curve (requires predicted probabilities)
- `--plot-threshold-analysis`: Plot threshold analysis (requires predicted probabilities)
- `--normalize-cm`: Normalize confusion matrix
- `--save-plots-dir`: Directory to save plot images

### compare
Compare ensemble vs individual models.

```bash
python fraud_cli.py compare --transactions <path> --identity <path> [options]
```

**Options:**
- `--plot-metrics-comparison`: Plot metrics comparison between models
- `--plot-confusion-matrices`: Plot confusion matrices for all models
- `--save-plots-dir`: Directory to save plot images

### quick-evaluate
Quick evaluation of data or model.

```bash
python fraud_cli.py quick-evaluate --transactions <path> --identity <path> [options]
```

**Options:**
- `--model`: Path to model file for evaluation (optional)
- `--plot-confusion-matrix`: Plot confusion matrix
- `--plot-roc-curve`: Plot ROC curve
- `--plot-pr-curve`: Plot Precision-Recall curve
- `--plot-threshold-analysis`: Plot threshold analysis
- `--normalize-cm`: Normalize confusion matrix
- `--save-plots-dir`: Directory to save plot images

### visualize
Generate visualizations for a trained model.

```bash
python fraud_cli.py visualize --model <path> --transactions <path> --identity <path> --save-plots-dir <dir> [options]
```

**Options:**
- `--plots`: Types of plots to generate: all, confusion_matrix, roc_curve, pr_curve, threshold_analysis, metrics_comparison, feature_importance (default: all)
- `--normalize-cm`: Normalize confusion matrix
- `--feature-names`: CSV file with feature names (optional)
- `--top-features`: Number of top features to show in feature importance plot (default: 20)

### interactive
Interactive command-line interface.

```bash
python fraud_cli.py interactive [--transactions <path>] [--identity <path>]
```

## Visualization Features

### Available Visualizations

1. **Confusion Matrix**
   - Shows true vs predicted labels
   - Optional normalization
   - Color-coded heatmap

2. **ROC Curve**
   - Receiver Operating Characteristic curve
   - Shows trade-off between true positive rate and false positive rate
   - Includes AUC (Area Under Curve) score
   *Requires predicted probabilities*

3. **Precision-Recall Curve**
   - Shows trade-off between precision and recall
   - Important for imbalanced datasets
   - Includes AP (Average Precision) score
   *Requires predicted probabilities*

4. **Threshold Analysis**
   - Shows how different probability thresholds affect precision, recall, and F1-score
   - Identifies optimal threshold
   *Requires predicted probabilities*

5. **Feature Importance**
   - Shows most important features for tree-based models
   - Shows coefficient magnitudes for linear models
   - Horizontal bar chart for easy comparison

6. **Metrics Comparison**
   - Compares accuracy, precision, recall, F1-score between models
   - Grouped bar chart for visual comparison

### Example Visualization Commands

```bash
# Generate all visualizations
python fraud_cli.py visualize --model model.pkl --transactions data.csv --identity id.csv --save-plots-dir visualizations/

# Generate specific plots
python fraud_cli.py visualize --model model.pkl --transactions data.csv --identity id.csv --save-plots-dir plots/ --plots confusion_matrix roc_curve feature_importance

# Test with visualizations
python fraud_cli.py test --model model.pkl --transactions test.csv --identity test_id.csv --plot-confusion-matrix --plot-roc-curve --save-plots-dir test_plots/
```

## ClassifierEvaluator API

The `ClassifierEvaluator` class in `models.py` provides comprehensive evaluation capabilities:

### Key Methods

```python
# Create evaluator
evaluator = ClassifierEvaluator(classifier_name, y_true, y_pred, y_pred_proba)

# Generate all visualizations
evaluator.visualize_all(save_dir="plots/")

# Individual plots
evaluator.plot_confusion_matrix(save_path="confusion_matrix.png")
evaluator.plot_roc_curve(save_path="roc_curve.png")
evaluator.plot_precision_recall_curve(save_path="pr_curve.png")
evaluator.plot_threshold_analysis(save_path="threshold_analysis.png")
evaluator.plot_feature_importance(model, feature_names, save_path="feature_importance.png")
evaluator.plot_metrics_comparison(other_evaluator, save_path="comparison.png")

# Comprehensive evaluation with visualizations
results = evaluator.evaluate(
    verbose=True,
    plot_confusion_matrix=True,
    plot_roc_curve=True,
    plot_pr_curve=True,
    plot_threshold_analysis=True
)
```

## Project Structure

```
├── models.py              # ML models and evaluation classes
├── fraud_cli.py          # Command-line interface
├── dataset.py            # Dataset handling
├── data.py               # Data loading utilities
├── preprocessor.py       # Data preprocessing
├── enhanced_pipeline.py  # Enhanced feature engineering pipelines
├── sample.py             # Data sampling utilities
├── cli_usage.md          # CLI usage documentation
├── demo_class_imbalance.py
├── enhanced_feature_engineer.py
├── enhanced_pipeline.py
├── test_*.py             # Test files
└── requirements.txt      # Python dependencies
```

## Development

### Running Tests
```bash
python test_cli.py
python test_auc_updates.py
python test_class_imbalance.py
```

### Adding New Features
1. Models are implemented in `models.py`
2. CLI commands are in `fraud_cli.py`
3. Data loading is in `data.py`
4. Preprocessing is in `preprocessor.py`

## License

This project is part of IT348 coursework.