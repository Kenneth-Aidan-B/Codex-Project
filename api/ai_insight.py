#!/usr/bin/env python3
"""
AI Insight endpoint for generating natural language summaries
of model predictions and SHAP explanations using Gemini LLM.

Environment Variables:
    GEMINI_API_KEY: API key for Gemini/Generative Language API
    GEMINI_MODEL: Model name (default: gemini-pro)
    GEMINI_ENDPOINT_TEMPLATE: API endpoint template (default: Google's API endpoint)
"""
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# Set up logging
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent.parent
EXPLANATIONS_DIR = BASE_DIR / "results" / "explanations"
AI_INSIGHTS_DIR = BASE_DIR / "results" / "ai_insights"

# Ensure AI insights directory exists
AI_INSIGHTS_DIR.mkdir(parents=True, exist_ok=True)

# Initialize router
router = APIRouter()


class InsightRequest(BaseModel):
    """Request model for AI insights"""
    sample_id: Optional[str] = Field(None, description="Sample ID to lookup SHAP explanation")
    probability: float = Field(..., description="Predicted probability (0-1)", ge=0, le=1)
    features: Optional[Dict[str, float]] = Field(None, description="Feature values")
    shap: Optional[Dict[str, float]] = Field(None, description="SHAP values")
    model_name: Optional[str] = Field("RandomForest", description="Model name")


class TopFeature(BaseModel):
    """Model for top feature with SHAP value"""
    feature: str
    shap_value: float


class InsightResponse(BaseModel):
    """Response model for AI insights"""
    model_used: str
    probability: float
    summary: str
    top_features: List[TopFeature]
    confidence_note: str
    next_step: str
    disclaimer: str
    llm_response_raw: Optional[str] = None
    cached: bool = False


def _call_gemini(prompt: str, temperature: float = 0.1, max_tokens: int = 500) -> Optional[str]:
    """
    Call Gemini API using simple requests-based HTTP call.
    
    NOTE: To use the official google-generativeai SDK instead:
    1. Install: pip install google-generativeai
    2. Replace this function with:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(
            prompt,
            generation_config={"temperature": temperature, "max_output_tokens": max_tokens}
        )
        return response.text
    
    Args:
        prompt: The prompt to send to Gemini
        temperature: Temperature for generation (default 0.1 for deterministic output)
        max_tokens: Maximum tokens to generate
    
    Returns:
        Generated text or None if call fails
    """
    # Read environment variables at call time (not module load time) for testability
    api_key = os.getenv("GEMINI_API_KEY", "")
    model_name = os.getenv("GEMINI_MODEL", "gemini-pro")
    endpoint_template = os.getenv(
        "GEMINI_ENDPOINT_TEMPLATE",
        "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    )
    
    if not api_key:
        logger.warning("GEMINI_API_KEY not set, skipping API call")
        return None
    
    try:
        # Format endpoint URL
        endpoint = endpoint_template.format(
            model=model_name,
            api_key=api_key
        )
        
        # Prepare request payload for Gemini API
        # Requesting JSON output in the prompt helps with parsing
        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "topP": 0.8,
                "topK": 40
            }
        }
        
        # Make API call
        response = requests.post(
            endpoint,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        # Check response
        if response.status_code == 200:
            result = response.json()
            # Extract text from Gemini response structure
            if "candidates" in result and len(result["candidates"]) > 0:
                candidate = result["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    parts = candidate["content"]["parts"]
                    if len(parts) > 0 and "text" in parts[0]:
                        return parts[0]["text"]
            logger.warning(f"Unexpected Gemini response structure: {result}")
            return None
        else:
            logger.error(f"Gemini API error: {response.status_code} - {response.text}")
            return None
    
    except Exception as e:
        logger.error(f"Error calling Gemini API: {e}")
        return None


def _build_fallback_summary(
    probability: float,
    shap_values: Optional[Dict[str, float]],
    features: Optional[Dict[str, float]],
    model_name: str
) -> str:
    """
    Build deterministic fallback summary when LLM is unavailable.
    
    Args:
        probability: Predicted probability
        shap_values: SHAP values for features
        features: Feature values
        model_name: Model name
    
    Returns:
        Deterministic summary string
    """
    risk_level = "high" if probability > 0.5 else "low"
    
    summary_parts = [
        f"The {model_name} model predicts a {risk_level} risk of hearing deficiency "
        f"with a probability of {probability:.2%}."
    ]
    
    if shap_values:
        # Sort by absolute SHAP value
        sorted_shap = sorted(
            shap_values.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]
        
        if sorted_shap:
            summary_parts.append("\nKey factors influencing this prediction:")
            for feature, value in sorted_shap:
                direction = "increases" if value > 0 else "decreases"
                summary_parts.append(f"- {feature}: {direction} risk (SHAP: {value:.4f})")
    
    if features:
        summary_parts.append(f"\nThis assessment is based on {len(features)} clinical and genomic features.")
    
    return " ".join(summary_parts)


def _build_prompt(
    sample_id: Optional[str],
    model_name: str,
    probability: float,
    top_shap: List[tuple],
    features: Optional[Dict[str, float]]
) -> str:
    """
    Build prompt for Gemini API.
    
    Args:
        sample_id: Sample identifier
        model_name: Model name
        probability: Predicted probability
        top_shap: Top SHAP features (name, value) tuples
        features: Feature values
    
    Returns:
        Formatted prompt string
    """
    risk_level = "HIGH" if probability > 0.5 else "LOW"
    
    prompt = f"""You are a clinical AI assistant explaining hearing deficiency risk predictions.

Sample ID: {sample_id or 'manual_input'}
Model: {model_name}
Predicted Risk: {risk_level} ({probability:.2%} probability)

Top Contributing Features (SHAP values):
"""
    
    for feature, value in top_shap:
        direction = "↑" if value > 0 else "↓"
        prompt += f"- {feature}: {direction} {abs(value):.4f}\n"
    
    prompt += """
Generate a concise clinical summary (2-3 sentences) in JSON format with these exact fields:
{
  "summary": "Brief explanation of the risk assessment",
  "key_factors": "Top 3 most important factors",
  "recommendation": "Suggested next step"
}

Requirements:
- Use clear, non-technical language appropriate for healthcare providers
- Focus on actionable insights
- Be concise but informative
- Return ONLY valid JSON, no other text
"""
    
    return prompt


def _load_shap_explanation(sample_id: str, model_name: str) -> Optional[Dict]:
    """
    Load SHAP explanation from file.
    
    Args:
        sample_id: Sample identifier
        model_name: Model name
    
    Returns:
        Dictionary with SHAP values or None if not found
    """
    # Try with model suffix first
    explanation_path = EXPLANATIONS_DIR / f"{sample_id}_{model_name}.json"
    
    if not explanation_path.exists():
        # Try without model suffix
        explanation_path = EXPLANATIONS_DIR / f"{sample_id}.json"
    
    if explanation_path.exists():
        try:
            with open(explanation_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading SHAP explanation: {e}")
    
    return None


def _cache_insight(cache_key: str, insight: Dict) -> None:
    """
    Cache generated insight to file.
    
    Args:
        cache_key: Cache key (sample_id or timestamp)
        insight: Insight data to cache
    """
    try:
        cache_path = AI_INSIGHTS_DIR / f"{cache_key}.json"
        with open(cache_path, 'w') as f:
            json.dump(insight, f, indent=2)
        logger.info(f"Cached insight to {cache_path}")
    except Exception as e:
        logger.error(f"Error caching insight: {e}")


def _load_cached_insight(cache_key: str) -> Optional[Dict]:
    """
    Load cached insight from file.
    
    Args:
        cache_key: Cache key (sample_id or timestamp)
    
    Returns:
        Cached insight or None if not found
    """
    try:
        cache_path = AI_INSIGHTS_DIR / f"{cache_key}.json"
        if cache_path.exists():
            with open(cache_path, 'r') as f:
                insight = json.load(f)
                logger.info(f"Loaded cached insight from {cache_path}")
                return insight
    except Exception as e:
        logger.error(f"Error loading cached insight: {e}")
    
    return None


@router.post("/insight", response_model=InsightResponse)
async def generate_insight(request: InsightRequest):
    """
    Generate AI-powered insight for model prediction.
    
    This endpoint accepts either:
    1. A sample_id to lookup pre-computed SHAP explanations
    2. Direct payload with probability, features, and SHAP values
    
    Environment Variables:
        - GEMINI_API_KEY: Required for LLM-powered summaries
        - GEMINI_MODEL: Model to use (default: gemini-pro)
        - GEMINI_ENDPOINT_TEMPLATE: Custom endpoint URL
    
    If GEMINI_API_KEY is not set or the API call fails, returns a
    deterministic fallback summary based on SHAP values.
    
    Generated insights are cached to reduce API calls.
    """
    try:
        # Determine cache key
        if request.sample_id:
            cache_key = request.sample_id
        else:
            # Generate unique key based on timestamp and request hash
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            request_hash = hash((
                request.probability,
                request.model_name,
                tuple(sorted(request.shap.items())) if request.shap else None,
                tuple(sorted(request.features.items())) if request.features else None
            ))
            cache_key = f"manual_{timestamp}_{abs(request_hash)}"
        
        # Check cache first
        cached = _load_cached_insight(cache_key)
        if cached:
            cached["cached"] = True
            # Convert top_features dicts back to TopFeature objects
            if "top_features" in cached and isinstance(cached["top_features"], list):
                cached["top_features"] = [
                    TopFeature(**f) if isinstance(f, dict) else f
                    for f in cached["top_features"]
                ]
            return InsightResponse(**cached)
        
        # Get SHAP values
        shap_values = request.shap
        features = request.features
        
        # If sample_id provided, try to load SHAP explanation
        if request.sample_id:
            explanation = _load_shap_explanation(request.sample_id, request.model_name)
            if explanation:
                shap_values = explanation.get("shap_values", {})
                # If top_features exists, we can use it
                if not shap_values and "top_features" in explanation:
                    # Convert top_features list to dict
                    shap_values = {feat[0]: feat[1] for feat in explanation["top_features"]}
        
        # Ensure we have SHAP values
        if not shap_values:
            raise HTTPException(
                status_code=400,
                detail="No SHAP values provided or found for sample_id"
            )
        
        # Get top SHAP features
        sorted_shap = sorted(
            shap_values.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:10]
        
        top_features = [
            TopFeature(feature=feat, shap_value=float(val))
            for feat, val in sorted_shap[:5]
        ]
        
        # Build prompt
        prompt = _build_prompt(
            request.sample_id,
            request.model_name,
            request.probability,
            sorted_shap[:5],
            features
        )
        
        # Call Gemini API
        llm_response = _call_gemini(prompt)
        
        # Build summary
        if llm_response:
            # Try to parse JSON from LLM response
            try:
                # Extract JSON from response (might have markdown formatting)
                json_start = llm_response.find('{')
                json_end = llm_response.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = llm_response[json_start:json_end]
                    llm_data = json.loads(json_str)
                    summary = llm_data.get("summary", "")
                    key_factors = llm_data.get("key_factors", "")
                    recommendation = llm_data.get("recommendation", "")
                    
                    # Combine into summary
                    summary_text = f"{summary} Key factors: {key_factors}"
                    next_step = recommendation
                else:
                    # Use raw response as summary
                    summary_text = llm_response
                    next_step = "Consult with healthcare provider for comprehensive evaluation."
            except json.JSONDecodeError:
                # Use raw response as summary
                summary_text = llm_response
                next_step = "Consult with healthcare provider for comprehensive evaluation."
        else:
            # Fallback to deterministic summary
            summary_text = _build_fallback_summary(
                request.probability,
                shap_values,
                features,
                request.model_name
            )
            next_step = "Consult with healthcare provider for comprehensive evaluation."
        
        # Build confidence note
        if request.probability > 0.7:
            confidence_note = "High confidence prediction"
        elif request.probability < 0.3:
            confidence_note = "High confidence prediction"
        else:
            confidence_note = "Moderate confidence - further assessment recommended"
        
        # Build response
        response_data = {
            "model_used": request.model_name,
            "probability": request.probability,
            "summary": summary_text,
            "top_features": top_features,
            "confidence_note": confidence_note,
            "next_step": next_step,
            "disclaimer": (
                "This AI-generated insight is for research and educational purposes only. "
                "NOT FOR CLINICAL USE. Do not send protected health information (PHI) to cloud LLM services. "
                "All predictions must be validated by qualified healthcare professionals."
            ),
            "llm_response_raw": llm_response if llm_response else None,
            "cached": False
        }
        
        # Cache the insight (convert TopFeature objects to dicts for JSON serialization)
        cache_data = {**response_data}
        cache_data["top_features"] = [
            {"feature": f.feature, "shap_value": f.shap_value}
            for f in top_features
        ]
        _cache_insight(cache_key, cache_data)
        
        return InsightResponse(**response_data)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating insight: {e}")
        raise HTTPException(status_code=500, detail=str(e))
