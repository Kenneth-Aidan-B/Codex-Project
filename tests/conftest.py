"""Pytest configuration and fixtures."""

import pytest
import tempfile
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_variant():
    """Sample variant data for testing."""
    return {
        "chrom": "chr13",
        "pos": 20763612,
        "ref": "C",
        "alt": "T",
        "gene": "GJB2",
        "clinvar_sig": "Pathogenic",
        "gnomad_af": 0.0001,
        "cadd_phred": 35.0,
        "consequence": "missense_variant",
        "genotype": "0/1"
    }


@pytest.fixture
def sample_analysis_result():
    """Sample analysis result for testing."""
    return {
        "sample_id": "TEST001",
        "risk_score": 0.75,
        "risk_category": "high",
        "confidence": 0.92,
        "key_findings": ["Pathogenic variant in GJB2"],
        "top_genes": [{
            "gene": "GJB2",
            "risk_score": 0.85,
            "variant_count": 1
        }],
        "recommendations": ["Genetic counseling recommended"]
    }


@pytest.fixture(autouse=True)
def mock_gemini_api(monkeypatch):
    """Mock Gemini API calls in all tests."""
    def mock_gemini(*args, **kwargs):
        return "Mock AI response"
    
    # This will automatically mock any Gemini API calls
    # Adjust the path based on actual implementation
    try:
        monkeypatch.setattr("api.ai_insight.get_gemini_response", mock_gemini)
    except:
        pass  # Gemini not imported, skip mocking
