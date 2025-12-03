"""Tests for report generation."""

import pytest
import json
import tempfile
from pathlib import Path
from reporting.report_generator import generate_clinical_report


def test_generate_json_report():
    """Test JSON report generation."""
    analysis_results = {
        "sample_id": "TEST001",
        "risk_score": 0.75,
        "risk_category": "high",
        "confidence": 0.92,
        "key_findings": ["Pathogenic variant in GJB2"],
        "recommendations": ["Genetic counseling recommended"]
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "report.json"
        
        result_path = generate_clinical_report(
            analysis_results,
            str(output_path),
            format="json"
        )
        
        assert Path(result_path).exists()
        
        with open(result_path) as f:
            report = json.load(f)
        
        assert report["sample_id"] == "TEST001"
        assert "risk_assessment" in report
        assert report["risk_assessment"]["risk_category"] == "high"


def test_generate_html_report():
    """Test HTML report generation."""
    analysis_results = {
        "sample_id": "TEST002",
        "risk_score": 0.3,
        "risk_category": "moderate",
        "confidence": 0.85,
        "key_findings": [],
        "recommendations": ["Standard follow-up"]
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "report.html"
        
        result_path = generate_clinical_report(
            analysis_results,
            str(output_path),
            format="html"
        )
        
        assert Path(result_path).exists()
        
        with open(result_path) as f:
            html = f.read()
        
        assert "TEST002" in html
        assert "moderate" in html.lower()
