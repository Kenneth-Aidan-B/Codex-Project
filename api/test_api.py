#!/usr/bin/env python3
"""
API tests for the hearing deficiency prediction API.
"""
import json
from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)


def test_root():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()


def test_models():
    """Test models list endpoint"""
    response = client.get("/models")
    assert response.status_code == 200
    assert "models" in response.json()


def test_predict():
    """Test prediction endpoint"""
    test_data = {
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
    
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200
    
    result = response.json()
    assert "risk_score" in result
    assert "prediction" in result
    assert "confidence" in result
    assert "sample_id" in result


if __name__ == "__main__":
    print("Running API tests...")
    test_root()
    print("✓ Root endpoint test passed")
    
    test_health()
    print("✓ Health check test passed")
    
    test_models()
    print("✓ Models list test passed")
    
    test_predict()
    print("✓ Prediction test passed")
    
    print("\n✓ All API tests passed!")
