#!/usr/bin/env python3
"""
Tests for the AI insight endpoint.
"""
import json
import os
from pathlib import Path
from unittest import mock

import pytest
from fastapi.testclient import TestClient

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.app import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def temp_ai_insights_dir(tmp_path, monkeypatch):
    """Use temporary directory for AI insights cache in all tests"""
    temp_insights_dir = tmp_path / "ai_insights"
    temp_insights_dir.mkdir(parents=True, exist_ok=True)
    
    import api.ai_insight
    monkeypatch.setattr(api.ai_insight, 'AI_INSIGHTS_DIR', temp_insights_dir)
    
    return temp_insights_dir


@pytest.fixture
def mock_gemini_success():
    """Mock successful Gemini API response"""
    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "candidates": [{
            "content": {
                "parts": [{
                    "text": json.dumps({
                        "summary": "The model predicts low risk of hearing deficiency based on favorable screening results.",
                        "key_factors": "Normal OAE and AABR results, healthy APGAR scores",
                        "recommendation": "Continue routine monitoring and follow-up"
                    })
                }]
            }
        }]
    }
    return mock_response


@pytest.fixture
def mock_gemini_error():
    """Mock failed Gemini API response"""
    mock_response = mock.Mock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    return mock_response


@pytest.fixture
def sample_shap_data():
    """Sample SHAP data for testing"""
    return {
        "sample_id": "test_sample_001",
        "model": "RandomForest",
        "shap_values": {
            "aabr_result": -0.071,
            "oae_result": -0.070,
            "apgar_5min": 0.058,
            "family_history_hearing_loss": -0.041,
            "syndromic_features": -0.033,
            "mechanical_ventilation_days": 0.021,
            "nicu_days": 0.020,
            "sex": -0.019,
            "birth_weight_g": -0.013,
            "premature": -0.012
        },
        "top_features": [
            ["aabr_result", -0.071],
            ["oae_result", -0.070],
            ["apgar_5min", 0.058],
            ["family_history_hearing_loss", -0.041],
            ["syndromic_features", -0.033]
        ]
    }


@pytest.fixture
def temp_explanations_dir(sample_shap_data, tmp_path):
    """Create temporary explanations directory with sample data"""
    explanations_dir = tmp_path / "results" / "explanations"
    explanations_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a sample explanation file
    explanation_file = explanations_dir / "test_sample_001_RandomForest.json"
    with open(explanation_file, 'w') as f:
        json.dump(sample_shap_data, f)
    
    # Also create one without model suffix
    explanation_file2 = explanations_dir / "test_sample_002.json"
    with open(explanation_file2, 'w') as f:
        json.dump(sample_shap_data, f)
    
    return explanations_dir


def test_insight_with_direct_payload_success(mock_gemini_success):
    """Test insight endpoint with direct payload and successful Gemini call"""
    with mock.patch('requests.post', return_value=mock_gemini_success), \
         mock.patch.dict(os.environ, {'GEMINI_API_KEY': 'test_api_key'}):
        
        request_data = {
            "probability": 0.15,
            "model_name": "RandomForest",
            "shap": {
                "aabr_result": -0.071,
                "oae_result": -0.070,
                "apgar_5min": 0.058,
                "family_history_hearing_loss": -0.041,
                "syndromic_features": -0.033
            },
            "features": {
                "aabr_result": 0,
                "oae_result": 0,
                "apgar_5min": 9
            }
        }
        
        response = client.post("/ai/insight", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        
        # Check required fields
        assert "model_used" in data
        assert data["model_used"] == "RandomForest"
        assert "probability" in data
        assert data["probability"] == 0.15
        assert "summary" in data
        assert len(data["summary"]) > 0
        assert "top_features" in data
        assert len(data["top_features"]) > 0
        assert "confidence_note" in data
        assert "next_step" in data
        assert "disclaimer" in data
        assert "llm_response_raw" in data
        assert "cached" in data
        
        # Verify disclaimer contains privacy warning
        assert "PHI" in data["disclaimer"] or "clinical use" in data["disclaimer"].lower()
        
        # Verify top_features structure
        for feature in data["top_features"]:
            assert "feature" in feature
            assert "shap_value" in feature


def test_insight_fallback_no_api_key():
    """Test fallback behavior when GEMINI_API_KEY is not set"""
    with mock.patch.dict(os.environ, {'GEMINI_API_KEY': ''}, clear=True):
        request_data = {
            "probability": 0.75,
            "model_name": "XGBoost",
            "shap": {
                "aabr_result": 0.08,
                "oae_result": 0.07,
                "mechanical_ventilation_days": 0.05,
                "nicu_days": 0.04,
                "family_history_hearing_loss": 0.03
            }
        }
        
        response = client.post("/ai/insight", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        
        # Should still return valid response with fallback summary
        assert "summary" in data
        assert "XGBoost" in data["summary"]
        assert "high risk" in data["summary"].lower()
        assert data["llm_response_raw"] is None  # No LLM response
        
        # Check that fallback includes SHAP information
        assert "aabr_result" in data["summary"] or "Key factors" in data["summary"]


def test_insight_fallback_api_error(mock_gemini_error):
    """Test fallback behavior when Gemini API returns error"""
    with mock.patch('requests.post', return_value=mock_gemini_error), \
         mock.patch.dict(os.environ, {'GEMINI_API_KEY': 'test_api_key'}):
        
        request_data = {
            "probability": 0.30,
            "model_name": "SVM",
            "shap": {
                "apgar_5min": -0.05,
                "birth_weight_g": -0.04,
                "gestational_age_weeks": -0.03
            }
        }
        
        response = client.post("/ai/insight", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        
        # Should use fallback summary
        assert "summary" in data
        assert "SVM" in data["summary"]
        assert data["llm_response_raw"] is None


def test_insight_with_sample_id(temp_explanations_dir, monkeypatch):
    """Test insight endpoint with sample_id lookup"""
    # Monkeypatch the EXPLANATIONS_DIR to use our temp directory
    import api.ai_insight
    monkeypatch.setattr(api.ai_insight, 'EXPLANATIONS_DIR', temp_explanations_dir)
    
    with mock.patch.dict(os.environ, {'GEMINI_API_KEY': ''}, clear=True):
        request_data = {
            "sample_id": "test_sample_001",
            "probability": 0.20,
            "model_name": "RandomForest"
        }
        
        response = client.post("/ai/insight", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        
        # Should successfully load SHAP from file
        assert "summary" in data
        assert "top_features" in data
        assert len(data["top_features"]) > 0
        
        # Verify that SHAP values were loaded
        feature_names = [f["feature"] for f in data["top_features"]]
        assert "aabr_result" in feature_names or "oae_result" in feature_names


def test_insight_with_sample_id_no_model_suffix(temp_explanations_dir, monkeypatch):
    """Test insight endpoint with sample_id lookup (no model suffix in filename)"""
    import api.ai_insight
    monkeypatch.setattr(api.ai_insight, 'EXPLANATIONS_DIR', temp_explanations_dir)
    
    with mock.patch.dict(os.environ, {'GEMINI_API_KEY': ''}, clear=True):
        request_data = {
            "sample_id": "test_sample_002",
            "probability": 0.45,
            "model_name": "XGBoost"
        }
        
        response = client.post("/ai/insight", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "summary" in data
        assert "top_features" in data


def test_insight_missing_shap_values():
    """Test error when no SHAP values provided and no sample_id"""
    request_data = {
        "probability": 0.50,
        "model_name": "RandomForest"
        # No shap or sample_id
    }
    
    response = client.post("/ai/insight", json=request_data)
    assert response.status_code == 400
    assert "SHAP" in response.json()["detail"]


def test_insight_caching():
    """Test that insights are cached properly"""
    with mock.patch.dict(os.environ, {'GEMINI_API_KEY': ''}, clear=True):
        request_data = {
            "sample_id": "cache_test_001",
            "probability": 0.60,
            "model_name": "RandomForest",
            "shap": {
                "feature1": 0.1,
                "feature2": -0.05
            }
        }
        
        # First call - should generate and cache
        response1 = client.post("/ai/insight", json=request_data)
        assert response1.status_code == 200
        data1 = response1.json()
        assert data1["cached"] is False
        
        # Second call - should return cached result
        response2 = client.post("/ai/insight", json=request_data)
        assert response2.status_code == 200
        data2 = response2.json()
        assert data2["cached"] is True
        
        # Content should be the same
        assert data1["summary"] == data2["summary"]
        assert data1["probability"] == data2["probability"]


def test_insight_probability_validation():
    """Test probability field validation"""
    # Test probability > 1
    request_data = {
        "probability": 1.5,
        "model_name": "RandomForest",
        "shap": {"feature1": 0.1}
    }
    
    response = client.post("/ai/insight", json=request_data)
    assert response.status_code == 422  # Validation error
    
    # Test probability < 0
    request_data["probability"] = -0.5
    response = client.post("/ai/insight", json=request_data)
    assert response.status_code == 422


def test_insight_confidence_levels():
    """Test different confidence levels based on probability"""
    test_cases = [
        (0.1, "high confidence"),  # Low probability -> high confidence
        (0.5, "moderate"),          # Medium probability -> moderate confidence
        (0.9, "high confidence")    # High probability -> high confidence
    ]
    
    with mock.patch.dict(os.environ, {'GEMINI_API_KEY': ''}, clear=True):
        for probability, expected_keyword in test_cases:
            request_data = {
                "probability": probability,
                "model_name": "RandomForest",
                "shap": {"feature1": 0.1, "feature2": -0.05}
            }
            
            response = client.post("/ai/insight", json=request_data)
            assert response.status_code == 200
            
            data = response.json()
            assert expected_keyword.lower() in data["confidence_note"].lower()


def test_insight_top_features_sorted():
    """Test that top features are sorted by absolute SHAP value"""
    with mock.patch.dict(os.environ, {'GEMINI_API_KEY': ''}, clear=True):
        request_data = {
            "probability": 0.40,
            "model_name": "RandomForest",
            "shap": {
                "feature_small": 0.01,
                "feature_large_negative": -0.10,
                "feature_large_positive": 0.08,
                "feature_medium": 0.05,
                "feature_tiny": -0.005,
                "feature_another": 0.07
            }
        }
        
        response = client.post("/ai/insight", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        top_features = data["top_features"]
        
        # Should have at most 5 features
        assert len(top_features) <= 5
        
        # Should be sorted by absolute SHAP value (descending)
        abs_values = [abs(f["shap_value"]) for f in top_features]
        assert abs_values == sorted(abs_values, reverse=True)
        
        # First feature should be the one with largest absolute value
        assert top_features[0]["feature"] == "feature_large_negative"


def test_gemini_api_call_structure(mock_gemini_success):
    """Test that Gemini API is called with correct structure"""
    with mock.patch('requests.post', return_value=mock_gemini_success) as mock_post, \
         mock.patch.dict(os.environ, {
             'GEMINI_API_KEY': 'test_key',
             'GEMINI_MODEL': 'gemini-pro'
         }):
        
        request_data = {
            "probability": 0.25,
            "model_name": "RandomForest",
            "shap": {"feature1": 0.1}
        }
        
        response = client.post("/ai/insight", json=request_data)
        assert response.status_code == 200
        
        # Verify Gemini API was called
        assert mock_post.called
        
        # Check call arguments
        call_args = mock_post.call_args
        
        # Verify URL contains API key
        assert 'test_key' in call_args[0][0]
        
        # Verify payload structure
        payload = call_args[1]['json']
        assert 'contents' in payload
        assert 'generationConfig' in payload
        assert payload['generationConfig']['temperature'] == 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
