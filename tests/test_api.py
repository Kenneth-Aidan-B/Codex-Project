#!/usr/bin/env python3
"""
Tests for the API using TestClient.
"""
import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.app import app

client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data


def test_health_endpoint():
    """Test health check"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"


def test_models_endpoint():
    """Test models list"""
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert isinstance(data["models"], list)


def test_predict_endpoint():
    """Test prediction with valid input"""
    test_input = {
        "age_months": 6,
        "sex": "M",
        "ethnicity": "Caucasian",
        "birth_weight_g": 3200,
        "gestational_age_weeks": 38.5,
        "premature": 0,
        "apgar_1min": 8,
        "apgar_5min": 9,
        "nicu_days": 0,
        "mechanical_ventilation_days": 0,
        "hyperbilirubinemia": 0,
        "bilirubin_max_mg_dl": 5.0,
        "ototoxic_medications": 0,
        "aminoglycoside_days": 0,
        "loop_diuretic_days": 0,
        "maternal_cmv_infection": 0,
        "maternal_rubella": 0,
        "maternal_toxoplasmosis": 0,
        "family_history_hearing_loss": 0,
        "consanguinity": 0,
        "syndromic_features": 0,
        "craniofacial_anomalies": 0,
        "oae_result": "pass",
        "aabr_result": "pass"
    }
    
    response = client.post("/predict", json=test_input)
    assert response.status_code == 200
    
    data = response.json()
    assert "sample_id" in data
    assert "risk_score" in data
    assert "prediction" in data
    assert "confidence" in data
    assert "model_used" in data
    
    # Check value ranges
    assert 0 <= data["risk_score"] <= 1
    assert 0 <= data["confidence"] <= 1
    assert data["prediction"] in ["Low Risk", "High Risk"]


def test_predict_high_risk():
    """Test prediction with high-risk input"""
    test_input = {
        "age_months": 3,
        "sex": "F",
        "ethnicity": "Asian",
        "birth_weight_g": 2000,
        "gestational_age_weeks": 32.0,
        "premature": 1,
        "apgar_1min": 4,
        "apgar_5min": 6,
        "nicu_days": 30,
        "mechanical_ventilation_days": 10,
        "hyperbilirubinemia": 1,
        "bilirubin_max_mg_dl": 22.0,
        "ototoxic_medications": 1,
        "aminoglycoside_days": 14,
        "loop_diuretic_days": 7,
        "maternal_cmv_infection": 1,
        "maternal_rubella": 0,
        "maternal_toxoplasmosis": 0,
        "family_history_hearing_loss": 1,
        "consanguinity": 1,
        "syndromic_features": 1,
        "craniofacial_anomalies": 1,
        "oae_result": "refer",
        "aabr_result": "refer"
    }
    
    response = client.post("/predict", json=test_input)
    assert response.status_code == 200
    
    data = response.json()
    # This should likely be high risk
    assert "risk_score" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
