# Fraud Detection CLI Usage Guide

This CLI provides a command-line interface for training and testing fraud detection models using the `SecurityFirstEnsemble` and `ClassifierEvaluator` from `models.py`.

THIS README WAS WRITTEN BY CLAUDE and reviewed by Jon Bailey for correctness.

## Installation

No special installation required. Just make sure you have the required dependencies:
```bash
pip install pandas scikit-learn matplotlib seaborn
```

## Available Commands

### 1. `train` - Train a new ensemble model

Train a SecurityFirstEnsemble model on fraud detection data.

```bash
python fraud_cli.py train \
  --transactions training/train_transaction.csv \
  --identity training/train_identity.csv \
  [--voting hard|soft|stacked] \
  [--tie-breaker malware|reject|confidence] \
  [--save-model model.pkl] \
  [--save-results results.json] \
  [--sample-fraction 0.1]
```

**Options:**
- `--transactions`: Path to transactions CSV (required)
- `--identity`: Path to identity CSV (required)
- `--voting`: Voting type (default: `hard`)
- `--tie-breaker`: Tie-breaking strategy (default: `malware`)
- `--save-model`: Save trained model to file
- `--save-results`: Save training results to JSON
- `--sample-fraction`: Use only a fraction of data (for quick testing)

### 2. `test` - Test a trained model

Evaluate a saved model on test data.

```bash
python fraud_cli.py test \
  --model model.pkl \
  --transactions testing/test_transaction.csv \
  --identity testing/test_identity.csv \
  [--threshold 0.5] \
  [--save-results test_results.json]
```

**Options:**
- `--model`: Path to trained model file (required)
- `--transactions`: Test transactions CSV (required)
- `--identity`: Test identity CSV (required)
- `--threshold`: Probability threshold for classification
- `--save-results`: Save test results to JSON

### 3. `compare` - Compare ensemble vs individual models

Train both ensemble and individual models and compare their performance.

```bash
python fraud_cli.py compare \
  --transactions training/train_transaction.csv \
  --identity training/train_identity.csv \
  [--voting hard|soft|stacked] \
  [--tie-breaker malware|reject|confidence] \
  [--save-results comparison.json]
```

### 4. `quick-evaluate` - Quick data/model evaluation

Get quick statistics about data or evaluate a model.

```bash
# Quick data statistics
python fraud_cli.py quick-evaluate \
  --transactions data.csv \
  --identity id.csv

# Evaluate a model
python fraud_cli.py quick-evaluate \
  --transactions data.csv \
  --identity id.csv \
  --model model.pkl \
  [--confusion-matrix] \
  [--plot-confusion-matrix]
```

## Examples

### Example 1: Basic Training
```bash
python fraud_cli.py train \
  --transactions training/train_transaction.csv \
  --identity training/train_identity.csv \
  --save-model my_model.pkl
```

### Example 2: Test with Custom Threshold
```bash
python fraud_cli.py test \
  --model my_model.pkl \
  --transactions testing/test_transaction.csv \
  --identity testing/test_identity.csv \
  --threshold 0.4 \
  --save-results test_output.json
```

### Example 3: Quick Data Analysis
```bash
python fraud_cli.py quick-evaluate \
  --transactions training/train_transaction.csv \
  --identity training/train_identity.csv
```

### Example 4: Compare Voting Strategies
```bash
# Compare hard vs soft voting
python fraud_cli.py compare \
  --transactions training/train_transaction.csv \
  --identity training/train_identity.csv \
  --voting hard \
  --save-results hard_voting.json

python fraud_cli.py compare \
  --transactions training/train_transaction.csv \
  --identity training/train_identity.csv \
  --voting soft \
  --save-results soft_voting.json
```

## Expected Data Format

The CLI expects two CSV files:

1. **Transactions CSV**: Contains transaction features
2. **Identity CSV**: Contains identity verification features

The files should merge on `TransactionID` column and the target variable should be named `isFraud`.

## Output Files

The CLI can generate several output files:

1. **Model files** (`.pkl`): Trained models saved using Python's `pickle` module
2. **JSON results**: Detailed metrics and evaluation results in JSON format
3. **Console output**: Detailed evaluation reports printed to console

## Running the Test Suite

A test script is provided to demonstrate functionality:

```bash
python test_cli.py
```

This will run through all major commands with a small sample of data.

## Tips

1. **For quick testing**: Use `--sample-fraction 0.01` to train on just 1% of data
2. **Save everything**: Use `--save-model` and `--save-results` to preserve your work
3. **Try different voting**: Experiment with `hard`, `soft`, and `stacked` voting types
4. **Visualization**: Use `--plot-confusion-matrix` with `quick-evaluate` to see visualizations

## Troubleshooting

**Issue**: "File not found" error
**Solution**: Make sure paths are correct. Use relative paths like `training/train_transaction.csv`

**Issue**: Memory errors with large datasets
**Solution**: Use `--sample-fraction` to train on a smaller subset

**Issue**: Missing dependencies
**Solution**: Install required packages: `pip install pandas scikit-learn`

## Integration with Existing Code

The CLI uses the same modules as your existing code:
- `models.py` for `SecurityFirstEnsemble` and `ClassifierEvaluator`
- `dataset.py` for `Dataset` class
- `sample.py` for `Sample` class
- `data.py` for `load_fraud_data`
- `preprocessor.py` for `FraudPreprocessor`

You can import and use these modules directly in Python code as well.