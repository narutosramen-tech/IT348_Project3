#!/usr/bin/env python3
"""
Test the updated ClassifierEvaluator with ROC-AUC and PR-AUC metrics.
"""

import numpy as np
from models import ClassifierEvaluator

def test_basic_evaluation():
    """Test basic evaluation without probabilities."""
    print("Test 1: Basic evaluation without probabilities")
    print("=" * 60)

    # Create simple test data
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1, 0, 1])

    evaluator = ClassifierEvaluator("TestClassifier", y_true, y_pred)
    results = evaluator.evaluate(verbose=True)

    print(f"\nMetrics: {results['metrics']}")
    assert 'roc_auc' in results['metrics'], "ROC-AUC should be in metrics (0 when no probabilities)"
    assert 'pr_auc' in results['metrics'], "PR-AUC should be in metrics (0 when no probabilities)"
    print("OK Basic evaluation test passed")

def test_evaluation_with_probabilities():
    """Test evaluation with predicted probabilities."""
    print("\n\nTest 2: Evaluation with predicted probabilities")
    print("=" * 60)

    # Create test data with probabilities
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1, 0, 1])
    y_pred_proba = np.array([0.1, 0.9, 0.2, 0.4, 0.1, 0.8, 0.3, 0.95])

    evaluator = ClassifierEvaluator("TestClassifier", y_true, y_pred, y_pred_proba)
    results = evaluator.evaluate(verbose=True)

    print(f"\nMetrics: {results['metrics']}")
    assert results['metrics']['roc_auc'] > 0, "ROC-AUC should be > 0 with probabilities"
    assert results['metrics']['pr_auc'] > 0, "PR-AUC should be > 0 with probabilities"
    print("OK Evaluation with probabilities test passed")

def test_comparison_with_probabilities():
    """Test comparison between two classifiers with probabilities."""
    print("\n\nTest 3: Comparison between classifiers")
    print("=" * 60)

    # Create two classifiers with different performance
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])

    # Classifier 1 (better)
    y_pred1 = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_pred_proba1 = np.array([0.1, 0.9, 0.2, 0.8, 0.1, 0.9, 0.2, 0.95])

    # Classifier 2 (worse)
    y_pred2 = np.array([0, 1, 0, 0, 0, 1, 0, 1])
    y_pred_proba2 = np.array([0.1, 0.7, 0.3, 0.4, 0.2, 0.6, 0.4, 0.8])

    evaluator1 = ClassifierEvaluator("BetterClassifier", y_true, y_pred1, y_pred_proba1)
    evaluator2 = ClassifierEvaluator("WorseClassifier", y_true, y_pred2, y_pred_proba2)

    print("\nComparing BetterClassifier vs WorseClassifier:")
    comparison = evaluator1.compare_with_other(evaluator2, verbose=True)

    # Should show BetterClassifier as overall winner
    print(f"\nOverall winner: {comparison['overall_winner']}")
    assert 'BetterClassifier' in comparison['overall_winner'], "BetterClassifier should win"
    print("OK Comparison test passed")

def test_comparison_without_probabilities():
    """Test comparison between two classifiers without probabilities."""
    print("\n\nTest 4: Comparison without probabilities")
    print("=" * 60)

    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])

    # Classifier 1
    y_pred1 = np.array([0, 1, 0, 1, 0, 1, 0, 1])

    # Classifier 2
    y_pred2 = np.array([0, 1, 0, 0, 0, 1, 0, 1])

    evaluator1 = ClassifierEvaluator("PerfectClassifier", y_true, y_pred1)
    evaluator2 = ClassifierEvaluator("GoodClassifier", y_true, y_pred2)

    print("\nComparing PerfectClassifier vs GoodClassifier:")
    comparison = evaluator1.compare_with_other(evaluator2, verbose=True)

    print(f"\nOverall winner: {comparison['overall_winner']}")
    print("OK Comparison without probabilities test passed")

def test_quick_evaluate_function():
    """Test the updated quick_evaluate_classifier function."""
    print("\n\nTest 5: Quick evaluate function")
    print("=" * 60)

    from models import quick_evaluate_classifier

    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1, 0, 1])
    y_pred_proba = np.array([0.1, 0.9, 0.2, 0.4, 0.1, 0.8, 0.3, 0.95])

    print("\nWithout probabilities:")
    results1 = quick_evaluate_classifier("TestClassifier", y_true, y_pred)

    print("\nWith probabilities:")
    results2 = quick_evaluate_classifier("TestClassifier", y_true, y_pred, y_pred_proba)

    assert results2['metrics']['roc_auc'] > results1['metrics']['roc_auc'], "ROC-AUC should be higher with probabilities"
    print("OK Quick evaluate function test passed")

def main():
    """Run all tests."""
    print("Testing Updated ClassifierEvaluator with ROC-AUC and PR-AUC")
    print("=" * 60)

    try:
        test_basic_evaluation()
        test_evaluation_with_probabilities()
        test_comparison_with_probabilities()
        test_comparison_without_probabilities()
        test_quick_evaluate_function()

        print("\n" + "=" * 60)
        print("All tests passed! OK")
        print("ClassifierEvaluator successfully updated with ROC-AUC and PR-AUC metrics.")

    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())