import os
import sys
import json
from fastapi.testclient import TestClient

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.app import app


client = TestClient(app)


def test_health_includes_gemini_enabled():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "gemini_enabled" in data
    assert isinstance(data["gemini_enabled"], bool)


def test_predict_includes_explanations():
    # Minimal input
    payload = {"pathogenic_variant_count": 1, "has_gjb2_variant": 1}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "patient_explanation" in data
    assert "clinician_explanation" in data
    # Ensure clinician explanation contains What/Why/How structure
    clinician = data["clinician_explanation"].lower()
    assert "what:" in clinician and "why:" in clinician and "how:" in clinician
