"""Tests for risk prediction."""

import pytest
from model.predictor import HearingLossPredictor, predict_sample_risk


def test_predictor_init():
    """Test predictor initialization."""
    predictor = HearingLossPredictor()
    assert predictor is not None
    assert predictor.gene_db is not None


def test_predict_risk_empty():
    """Test prediction with no variants."""
    predictor = HearingLossPredictor()
    
    prediction = predictor.predict_risk([], None)
    
    assert prediction.overall_risk_score == 0.0
    assert prediction.risk_category == "low"


def test_predict_risk_pathogenic():
    """Test prediction with pathogenic variant."""
    predictor = HearingLossPredictor()
    
    variants = [{
        "gene": "GJB2",
        "chrom": "chr13",
        "pos": 20763612,
        "ref": "C",
        "alt": "T",
        "clinvar_sig": "Pathogenic",
        "gnomad_af": 0.0001,
        "cadd_phred": 35.0,
        "consequence": "missense_variant",
        "genotype": "1/1"
    }]
    
    prediction = predictor.predict_risk(variants, {"sample_id": "TEST001"})
    
    assert prediction.sample_id == "TEST001"
    assert prediction.overall_risk_score > 0.0
    assert len(prediction.variant_scores) == 1
    assert len(prediction.gene_scores) > 0


def test_predict_sample_risk():
    """Test convenience function for prediction."""
    variants = [{
        "gene": "SLC26A4",
        "clinvar_sig": "Likely pathogenic",
        "gnomad_af": 0.0005,
        "cadd_phred": 28.0,
        "consequence": "frameshift_variant"
    }]
    
    result = predict_sample_risk(variants, "SAMPLE001")
    
    assert "sample_id" in result
    assert "risk_score" in result
    assert "risk_category" in result
    assert result["sample_id"] == "SAMPLE001"
