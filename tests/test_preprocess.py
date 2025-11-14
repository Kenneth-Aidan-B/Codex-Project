#!/usr/bin/env python3
"""
Tests for preprocessing pipeline.
"""
import pytest
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def test_processed_features_exists():
    """Test that processed_features.csv exists"""
    assert (PROCESSED_DIR / "processed_features.csv").exists(), "processed_features.csv not found"


def test_preprocessing_artifacts_exist():
    """Test that preprocessing artifacts exist"""
    assert (PROCESSED_DIR / "preprocessing_artifacts.joblib").exists(), "preprocessing_artifacts.joblib not found"


def test_metadata_exists():
    """Test that metadata.json exists"""
    assert (PROCESSED_DIR / "metadata.json").exists(), "metadata.json not found"


def test_processed_features_structure():
    """Test processed features structure"""
    df = pd.read_csv(PROCESSED_DIR / "processed_features.csv")
    
    assert 'sample_id' in df.columns, "Missing sample_id column"
    assert 'hearing_impairment' in df.columns, "Missing hearing_impairment column"
    assert len(df) > 0, "processed_features.csv is empty"


def test_smote_applied():
    """Test that SMOTE was applied (balanced classes)"""
    df = pd.read_csv(PROCESSED_DIR / "processed_features.csv")
    
    value_counts = df['hearing_impairment'].value_counts()
    
    # After SMOTE, classes should be balanced
    assert len(value_counts) == 2, "Should have 2 classes"
    
    # Check balance (allow some tolerance)
    ratio = value_counts.min() / value_counts.max()
    assert ratio > 0.8, f"Classes not well balanced: {ratio:.2f}"


def test_no_missing_values():
    """Test that there are no missing values after preprocessing"""
    df = pd.read_csv(PROCESSED_DIR / "processed_features.csv")
    
    # Exclude sample_id from check
    data_cols = [col for col in df.columns if col != 'sample_id']
    
    missing_counts = df[data_cols].isnull().sum()
    assert missing_counts.sum() == 0, f"Found missing values: {missing_counts[missing_counts > 0]}"


def test_scaled_features():
    """Test that numerical features are scaled"""
    df = pd.read_csv(PROCESSED_DIR / "processed_features.csv")
    
    # Select numerical columns (exclude sample_id and target)
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    numerical_cols = [col for col in numerical_cols if col not in ['sample_id', 'hearing_impairment']]
    
    if len(numerical_cols) > 0:
        # After scaling, most features should have mean ~ 0 and std ~ 1
        # But binary features will have std ~ 0.5, so we need to filter those out
        for col in numerical_cols[:5]:  # Check first 5
            # Skip binary features (only 2 unique values after encoding)
            if df[col].nunique() <= 2:
                continue
            
            mean = df[col].mean()
            std = df[col].std()
            # Allow some tolerance
            assert abs(mean) < 1.0, f"Feature {col} not centered: mean={mean:.2f}"
            assert 0.5 < std < 2.0, f"Feature {col} not scaled: std={std:.2f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
