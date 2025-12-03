"""Tests for database models."""

import pytest
from datetime import datetime
from database.models import Sample, Analysis, Variant, Report, AuditLog


def test_sample_model():
    """Test Sample ORM model."""
    sample = Sample(
        external_id="TEST001",
        sex="M",
        status="received"
    )
    
    assert sample.external_id == "TEST001"
    assert sample.sex == "M"
    assert sample.status == "received"


def test_analysis_model():
    """Test Analysis ORM model."""
    analysis = Analysis(
        sample_id=1,
        model_version="v1.0",
        risk_score=0.75,
        risk_category="high",
        confidence=0.92,
        status="completed"
    )
    
    assert analysis.risk_score == 0.75
    assert analysis.risk_category == "high"


def test_variant_model():
    """Test Variant ORM model."""
    variant = Variant(
        analysis_id=1,
        chromosome="chr13",
        position=20763612,
        ref="C",
        alt="T",
        gene="GJB2",
        consequence="missense_variant",
        clinvar_sig="Pathogenic",
        gnomad_af=0.0001,
        pathogenicity_score=0.9
    )
    
    assert variant.gene == "GJB2"
    assert variant.pathogenicity_score == 0.9
