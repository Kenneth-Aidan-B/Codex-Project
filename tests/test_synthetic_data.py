#!/usr/bin/env python3
"""
Tests for synthetic data generation.
"""
import pytest
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "synthetic"


def test_variants_csv_exists():
    """Test that variants.csv exists"""
    assert (DATA_DIR / "variants.csv").exists(), "variants.csv not found"


def test_clinical_csv_exists():
    """Test that clinical.csv exists"""
    assert (DATA_DIR / "clinical.csv").exists(), "clinical.csv not found"


def test_features_csv_exists():
    """Test that features.csv exists"""
    assert (DATA_DIR / "features.csv").exists(), "features.csv not found"


def test_variants_structure():
    """Test variants.csv structure"""
    df = pd.read_csv(DATA_DIR / "variants.csv")
    
    required_columns = ['sample_id', 'gene', 'variant_id', 'chromosome', 'position',
                       'ref', 'alt', 'variant_type', 'zygosity', 'pathogenicity']
    
    for col in required_columns:
        assert col in df.columns, f"Missing column: {col}"
    
    assert len(df) > 0, "variants.csv is empty"


def test_clinical_structure():
    """Test clinical.csv structure"""
    df = pd.read_csv(DATA_DIR / "clinical.csv")
    
    required_columns = ['sample_id', 'age_months', 'sex', 'ethnicity',
                       'birth_weight_g', 'gestational_age_weeks',
                       'hearing_impairment']
    
    for col in required_columns:
        assert col in df.columns, f"Missing column: {col}"
    
    assert len(df) >= 1000, f"Expected >= 1000 samples, got {len(df)}"


def test_features_structure():
    """Test features.csv structure"""
    df = pd.read_csv(DATA_DIR / "features.csv")
    
    assert 'sample_id' in df.columns
    assert 'hearing_impairment' in df.columns
    assert len(df) >= 1000, f"Expected >= 1000 samples, got {len(df)}"


def test_target_distribution():
    """Test that target variable has reasonable distribution"""
    df = pd.read_csv(DATA_DIR / "features.csv")
    
    positive_rate = df['hearing_impairment'].mean()
    assert 0.10 <= positive_rate <= 0.50, f"Unusual positive class rate: {positive_rate:.2%}"


def test_no_duplicate_sample_ids():
    """Test that sample IDs are unique"""
    df = pd.read_csv(DATA_DIR / "features.csv")
    
    assert df['sample_id'].nunique() == len(df), "Duplicate sample IDs found"


def test_value_ranges():
    """Test that values are in reasonable ranges"""
    df = pd.read_csv(DATA_DIR / "features.csv")
    
    assert df['birth_weight_g'].min() >= 500, "Birth weight too low"
    assert df['birth_weight_g'].max() <= 6000, "Birth weight too high"
    
    assert df['gestational_age_weeks'].min() >= 20, "Gestational age too low"
    assert df['gestational_age_weeks'].max() <= 45, "Gestational age too high"
    
    assert df['apgar_1min'].min() >= 0, "APGAR score out of range"
    assert df['apgar_1min'].max() <= 10, "APGAR score out of range"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
