#!/usr/bin/env python3
"""
Test script to verify that model_utils functions work correctly.
This script tests the creation of all supported classifiers.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))

from models.model_utils import ModelBuilder, create_classifier

def test_individual_classifiers():
    """Test creating individual classifiers."""
    print("Testing individual classifier creation...")
    
    # Create some sample data
    X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    
    classifier_names = ModelBuilder.get_classifier_names()
    
    for name in classifier_names:
        try:
            print(f"  Testing {name}...")
            model = create_classifier(name)
            
            # Test fitting and prediction
            model.fit(X, y)
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)
            
            print(f"    ✓ {name}: Shape {predictions.shape}, Proba shape {probabilities.shape}")
            
        except Exception as e:
            print(f"    ✗ {name}: Failed with error {e}")
    
    print("Individual classifier test completed.\n")

def test_get_all_classifiers():
    """Test getting all classifiers at once."""
    print("Testing get_all_classifiers function...")
    
    try:
        # Test with default parameters
        classifiers = ModelBuilder.get_all_classifiers()
        print(f"  ✓ Created {len(classifiers)} classifiers with default params")
        
        # Test with custom parameters
        custom_params = {
            'random_forest': {'n_estimators': 50},
            'logistic_regression': {'max_iter': 500}
        }
        classifiers_custom = ModelBuilder.get_all_classifiers(config_params=custom_params)
        print(f"  ✓ Created {len(classifiers_custom)} classifiers with custom params")
        
        # Test with class weights disabled
        classifiers_no_weights = ModelBuilder.get_all_classifiers(use_class_weights=False)
        print(f"  ✓ Created {len(classifiers_no_weights)} classifiers without class weights")
        
    except Exception as e:
        print(f"  ✗ get_all_classifiers failed: {e}")
    
    print("get_all_classifiers test completed.\n")

def test_classifier_training():
    """Test training a few classifiers on sample data."""
    print("Testing classifier training...")
    
    # Create sample data
    X, y = make_classification(n_samples=200, n_features=20, n_classes=2, 
                             n_informative=10, n_redundant=5, random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    
    # Test a few representative classifiers
    test_models = ['random_forest', 'logistic_regression', 'svm_rbf']
    
    for name in test_models:
        try:
            print(f"  Testing training for {name}...")
            model = create_classifier(name)
            
            # Split data
            split_idx = int(0.8 * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Train and evaluate
            model.fit(X_train, y_train)
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            print(f"    ✓ {name}: Train score {train_score:.3f}, Test score {test_score:.3f}")
            
        except Exception as e:
            print(f"    ✗ {name}: Training failed with error {e}")
    
    print("Classifier training test completed.\n")

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Model Utils Functions")
    print("=" * 60)
    
    test_individual_classifiers()
    test_get_all_classifiers()
    test_classifier_training()
    
    print("All tests completed!")

if __name__ == "__main__":
    main() 