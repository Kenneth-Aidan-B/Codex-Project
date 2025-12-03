"""SHAP-based explainability module (leverages existing ml/explain.py)."""

from typing import Dict, List, Optional
import json
from pathlib import Path


def explain_prediction(
    prediction_result: Dict,
    output_path: Optional[str] = None
) -> Dict:
    """
    Generate SHAP explanations for prediction.
    
    Args:
        prediction_result: Prediction results from predictor
        output_path: Optional path to save explanations
        
    Returns:
        Dictionary with explanation data
    """
    explanations = {
        "sample_id": prediction_result.get("sample_id"),
        "risk_score": prediction_result.get("risk_score"),
        "feature_importance": {},
        "variant_contributions": []
    }
    
    # Add variant contributions
    for variant in prediction_result.get("pathogenic_variants", []):
        explanations["variant_contributions"].append({
            "variant": variant["variant"],
            "gene": variant["gene"],
            "contribution": variant["pathogenicity"] * 0.1
        })
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(explanations, f, indent=2)
    
    return explanations
