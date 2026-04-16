#!/usr/bin/env python3
"""
Test script for the Fraud Detection CLI

This script demonstrates how to use the fraud_cli.py command-line interface
with the provided dataset files.
"""

import subprocess
import sys
import os

def run_command(command):
    """Run a CLI command and print the output."""
    print(f"\n{'='*60}")
    print(f"Running: python fraud_cli.py {command}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            f"python fraud_cli.py {command}",
            shell=True,
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
        if result.stderr:
            print(f"STDERR:\n{result.stderr}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        print(f"STDOUT:\n{e.stdout}")
        print(f"STDERR:\n{e.stderr}")
        return False

def main():
    print("Testing Fraud Detection CLI")
    print("=" * 60)

    # Define paths to data files
    train_transactions = "training/train_transaction.csv"
    train_identity = "training/train_identity.csv"
    test_transactions = "testing/test_transaction.csv"
    test_identity = "testing/test_identity.csv"

    # Check if data files exist
    if not os.path.exists(train_transactions):
        print(f"Error: Training transactions file not found: {train_transactions}")
        return
    if not os.path.exists(train_identity):
        print(f"Error: Training identity file not found: {train_identity}")
        return

    print(f"Data files found:")
    print(f"  Training transactions: {train_transactions}")
    print(f"  Training identity: {train_identity}")

    if os.path.exists(test_transactions) and os.path.exists(test_identity):
        print(f"  Test transactions: {test_transactions}")
        print(f"  Test identity: {test_identity}")

    print("\n" + "=" * 60)

    # Test 1: Quick evaluation of training data
    print("\nTest 1: Quick evaluation of training data")
    success1 = run_command(f"quick-evaluate --transactions {train_transactions} --identity {train_identity}")

    # Test 2: Train a model (use small sample for quick testing)
    print("\nTest 2: Training a model with small sample")
    success2 = run_command(f"train --transactions {train_transactions} --identity {train_identity} --sample-fraction 0.01 --save-model test_model.pkl --save-results training_results.json")

    # Test 3: Compare ensemble vs individual models
    if success2:
        print("\nTest 3: Comparing ensemble vs individual models")
        success3 = run_command(f"compare --transactions {train_transactions} --identity {train_identity} --sample-fraction 0.01 --save-results comparison_results.json")

    # Test 4: Test the trained model if test data exists
    if os.path.exists(test_transactions) and os.path.exists(test_identity):
        if success2 and os.path.exists("test_model.pkl"):
            print("\nTest 4: Testing trained model on test data")
            success4 = run_command(f"test --model test_model.pkl --transactions {test_transactions} --identity {test_identity} --save-results test_results.json")

    print("\n" + "=" * 60)
    print("Testing complete!")
    print("Generated files:")

    files_to_check = [
        "test_model.pkl",
        "training_results.json",
        "comparison_results.json",
        "test_results.json"
    ]

    for file in files_to_check:
        if os.path.exists(file):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} (not created)")

    print("\nUsage examples:")
    print("  # Train a full model:")
    print(f"  python fraud_cli.py train --transactions {train_transactions} --identity {train_identity} --save-model full_model.pkl")
    print("\n  # Test a model:")
    print(f"  python fraud_cli.py test --model full_model.pkl --transactions {test_transactions} --identity {test_identity}")
    print("\n  # Quick evaluation:")
    print(f"  python fraud_cli.py quick-evaluate --transactions {train_transactions} --identity {train_identity}")

if __name__ == "__main__":
    main()