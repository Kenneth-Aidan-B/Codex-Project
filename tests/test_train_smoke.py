#!/usr/bin/env python3
"""
Smoke tests for model training.
"""
import pytest
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"


def test_metrics_file_exists():
    """Test that metrics.csv was created"""
    assert (RESULTS_DIR / "metrics.csv").exists(), "metrics.csv not found"


def test_models_saved():
    """Test that models were saved"""
    expected_models = ['RandomForest', 'SVM', 'XGBoost', 'ANN']
    
    for model_name in expected_models:
        model_dir = MODELS_DIR / model_name
        assert model_dir.exists(), f"{model_name} directory not found"
        
        # Check for latest model
        latest_model = model_dir / "model_latest.joblib"
        if not latest_model.exists():
            # For neural network models, check for .pt files
            latest_model_pt = model_dir / "model_latest.pt"
            assert latest_model.exists() or latest_model_pt.exists(), \
                f"No model file found for {model_name}"


def test_metrics_content():
    """Test metrics file content"""
    import pandas as pd
    
    df = pd.read_csv(RESULTS_DIR / "metrics.csv")
    
    required_columns = ['model', 'accuracy', 'precision', 'recall', 'f1', 'auc']
    for col in required_columns:
        assert col in df.columns, f"Missing column: {col}"
    
    # Check that metrics are in valid ranges
    for col in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        assert (df[col] >= 0).all(), f"{col} has negative values"
        assert (df[col] <= 1).all(), f"{col} has values > 1"


def test_at_least_one_model():
    """Test that at least one model achieved reasonable performance"""
    import pandas as pd
    
    df = pd.read_csv(RESULTS_DIR / "metrics.csv")
    
    # At least one model should have AUC > 0.6
    assert (df['auc'] > 0.6).any(), "No model achieved AUC > 0.6"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
