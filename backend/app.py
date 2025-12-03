import json
import os
from typing import Any, Dict, List, Optional

try:
    import google.generativeai as genai
except Exception:
    genai = None
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-2.0-flash"
gemini_client = None
gemini_api_key = os.getenv("GEMINI_API_KEY")
GEMINI_ENABLED = False
if gemini_api_key and genai is not None:
    try:
        genai.configure(api_key=gemini_api_key)
        # The SDK can expose a model wrapper; create if available
        try:
            gemini_client = genai.GenerativeModel(GEMINI_MODEL)
        except Exception:
            gemini_client = None
        logger.info("Gemini client configured")
    except Exception as exc:  # pragma: no cover - configuration errors logged and ignored
        logger.warning("Failed to configure Gemini client: %s", exc)

    GEMINI_ENABLED = bool(gemini_client)

app = FastAPI(
    title="Genetic Hearing Risk Prediction API",
    description=(
        "API for predicting hearing deficiency risk based purely on genetic data. "
        "This system analyzes genomic variants and genetic risk factors only. "
        "For research and demonstration purposes - not for clinical use."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Genetic-only request model
class GeneticPredictRequest(BaseModel):
    """Request model for genetic-only prediction"""
    sample_id: Optional[str] = None
    
    # Genetic profile features
    ethnicity: str = Field(default="Caucasian", description="Ethnicity")
    family_history_hearing_loss: int = Field(default=0, ge=0, le=1)
    consanguinity: int = Field(default=0, ge=0, le=1)
    syndromic_genetic_condition: int = Field(default=0, ge=0, le=1)
    mtdna_variant_detected: int = Field(default=0, ge=0, le=1)
    polygenic_risk_score: float = Field(default=0.5, ge=0.0, le=1.0)
    gjb2_carrier: int = Field(default=0, ge=0, le=1)
    num_affected_relatives: int = Field(default=0, ge=0, le=10)
    
    # Variant counts
    total_variant_count: int = Field(default=0, ge=0)
    pathogenic_variant_count: int = Field(default=0, ge=0)
    likely_pathogenic_count: int = Field(default=0, ge=0)
    vus_count: int = Field(default=0, ge=0)
    benign_variant_count: int = Field(default=0, ge=0)
    
    # Gene-specific variants
    has_gjb2_variant: int = Field(default=0, ge=0, le=1)
    has_slc26a4_variant: int = Field(default=0, ge=0, le=1)
    has_otof_variant: int = Field(default=0, ge=0, le=1)
    has_myo7a_variant: int = Field(default=0, ge=0, le=1)
    has_cdh23_variant: int = Field(default=0, ge=0, le=1)
    has_tmc1_variant: int = Field(default=0, ge=0, le=1)
    
    # Zygosity and impact
    homozygous_variant_count: int = Field(default=0, ge=0)
    compound_het_count: int = Field(default=0, ge=0)
    high_impact_count: int = Field(default=0, ge=0)
    moderate_impact_count: int = Field(default=0, ge=0)
    
    # Variant scores
    max_cadd_score: float = Field(default=0.0, ge=0.0, le=50.0)
    mean_cadd_score: float = Field(default=0.0, ge=0.0, le=50.0)
    max_revel_score: float = Field(default=0.0, ge=0.0, le=1.0)
    rare_variant_count: int = Field(default=0, ge=0)
    unique_genes_affected: int = Field(default=0, ge=0)


class GeneticPredictionResponse(BaseModel):
    sample_id: Optional[str]
    genetic_risk_score: float
    risk_category: str
    confidence: float
    prediction: str
    top_features: List[Dict[str, Any]]
    patient_explanation: str
    clinician_explanation: str
    disclaimer: str = "Genetic analysis prototype. Not for clinical use."


def calculate_genetic_risk(features: Dict[str, Any]) -> float:
    """Calculate genetic risk score based on variant features"""
    risk = 0.0
    
    risk += features.get('pathogenic_variant_count', 0) * 0.25
    risk += features.get('likely_pathogenic_count', 0) * 0.15
    risk += features.get('homozygous_variant_count', 0) * 0.20
    risk += features.get('compound_het_count', 0) * 0.18
    risk += features.get('high_impact_count', 0) * 0.12
    risk += features.get('has_gjb2_variant', 0) * 0.15
    risk += features.get('has_slc26a4_variant', 0) * 0.10
    risk += features.get('has_myo7a_variant', 0) * 0.12
    risk += features.get('has_cdh23_variant', 0) * 0.12
    risk += features.get('has_otof_variant', 0) * 0.10
    risk += features.get('has_tmc1_variant', 0) * 0.08
    risk += (features.get('max_cadd_score', 0) / 40) * 0.10
    risk += features.get('max_revel_score', 0) * 0.08
    risk += features.get('family_history_hearing_loss', 0) * 0.15
    risk += features.get('consanguinity', 0) * 0.12
    risk += features.get('syndromic_genetic_condition', 0) * 0.20
    risk += features.get('mtdna_variant_detected', 0) * 0.15
    risk += features.get('num_affected_relatives', 0) * 0.05
    risk += features.get('polygenic_risk_score', 0.5) * 0.10
    
    return min(max(risk, 0.0), 1.0)


def get_top_contributing_features(features: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Calculate contribution of each genetic feature"""
    contributions = []
    
    feature_weights = {
        'pathogenic_variant_count': ('Pathogenic Variants', 0.25),
        'likely_pathogenic_count': ('Likely Pathogenic', 0.15),
        'homozygous_variant_count': ('Homozygous Variants', 0.20),
        'compound_het_count': ('Compound Het', 0.18),
        'high_impact_count': ('High Impact', 0.12),
        'has_gjb2_variant': ('GJB2 Variant', 0.15),
        'has_slc26a4_variant': ('SLC26A4 Variant', 0.10),
        'has_myo7a_variant': ('MYO7A Variant', 0.12),
        'has_cdh23_variant': ('CDH23 Variant', 0.12),
        'has_otof_variant': ('OTOF Variant', 0.10),
        'has_tmc1_variant': ('TMC1 Variant', 0.08),
        'family_history_hearing_loss': ('Family History', 0.15),
        'consanguinity': ('Consanguinity', 0.12),
        'syndromic_genetic_condition': ('Syndromic', 0.20),
        'mtdna_variant_detected': ('mtDNA Variant', 0.15),
    }
    
    for key, (name, weight) in feature_weights.items():
        value = features.get(key, 0)
        contribution = value * weight
        if contribution > 0:
            contributions.append({
                'feature': name,
                'value': value,
                'contribution': round(contribution, 4)
            })
    
    if features.get('max_cadd_score', 0) > 15:
        contributions.append({
            'feature': 'CADD Score',
            'value': features.get('max_cadd_score', 0),
            'contribution': round((features.get('max_cadd_score', 0) / 40) * 0.10, 4)
        })
    
    if features.get('max_revel_score', 0) > 0.5:
        contributions.append({
            'feature': 'REVEL Score',
            'value': features.get('max_revel_score', 0),
            'contribution': round(features.get('max_revel_score', 0) * 0.08, 4)
        })
    
    contributions = sorted(contributions, key=lambda x: x['contribution'], reverse=True)[:5]
    total = sum(c['contribution'] for c in contributions)
    for c in contributions:
        c['importance'] = round(c['contribution'] / max(total, 0.001), 4)
    
    return contributions


def _default_patient_summary(risk_category: str, risk_score: float) -> str:
    # Compact patient-facing summary with brief next steps
    return (
        f"What this means: Your genetic risk is {risk_category} ({risk_score*100:.1f}%). "
        "Why this matters: genetic variants associated with hearing loss were detected. "
        "Next steps: consider genetic counseling and confirmatory testing with a specialist."
    )


def _default_clinician_summary(top_features: List[Dict[str, Any]]) -> str:
    # Return a compact structured clinician summary: What / Why / How
    if not top_features:
        return "What: No dominant genetic contributors detected. Why: No high-impact variants or significant counts. How: Consider comprehensive genomic testing if clinical suspicion remains."
    highlights = ", ".join(f"{f['feature']} ({f['contribution']:.2f})" for f in top_features)
    what = f"What: Elevated genetic risk driven mainly by {highlights}."
    why = "Why: Pathogenic and likely pathogenic variants increase risk, especially when homozygous or compound heterozygous in key hearing genes."
    how = "How: Recommend targeted variant confirmation, family segregation testing, referrals to a genetic counselor, and consideration for clinical audiology follow-up."
    return f"{what} {why} {how}"


def format_patient_summary(risk_category: str, risk_score: float, top_features: List[Dict[str, Any]]) -> str:
    """Guaranteed concise patient-friendly summary used as fallback or post-processing.
    Keep it empathetic and ≤ 120 words.
    """
    main_features = ", ".join([f["feature"] for f in top_features[:3]]) if top_features else "no specific variants"
    return (
        f"What this means: Your genetic risk is {risk_category} ({risk_score*100:.1f}%). "
        f"Key findings: {main_features}. "
        "Next steps: consider genetic counseling and confirmatory testing to clarify these results."
    )


def format_clinician_summary(top_features: List[Dict[str, Any]], risk_category: str, risk_score: float, features: Dict[str, Any]) -> str:
    """Format a concise clinician-focused summary including What, Why, How for immediate clinical context.
    Keep it structured and direct (short bullet-like sentences).
    """
    if not top_features:
        what = "What: No dominant genetic contributors detected."
        why = "Why: No high-impact variants or relevant allele frequency signals in the submitted profile."
        how = "How: If clinically indicated, order expanded sequencing or CNV testing; refer to genetics as appropriate."
        return f"{what} {why} {how}"

    top = top_features[:3]
    top_str = ", ".join(f"{t['feature']}({t['contribution']:.2f})" for t in top)
    what = f"What: Primary drivers - {top_str}. Risk category: {risk_category} ({risk_score*100:.1f}%)."
    why = "Why: Accumulation of pathogenic/likely-pathogenic variants and elevated predicted impact scores (CADD/REVEL) increases the probability of genetically mediated hearing loss."
    how = "How: Confirm variants by orthogonal methods, consider segregation, counsel patient on recurrence risk, and coordinate audiologic testing and genetics referral."
    return f"{what} {why} {how}"


def generate_ai_insights(context: Dict[str, Any]) -> Dict[str, str]:
    """Leverage Gemini to build patient- and clinician-focused summaries."""
    if not gemini_client:
        return {
            "patient": _default_patient_summary(context["risk_category"], context["risk_score"]),
            "clinician": _default_clinician_summary(context["top_features"]),
        }

    prompt = (
        "You are assisting with a genetics-only hearing risk report. "
        "Return JSON with keys 'patient_summary' and 'clinician_summary'. "
        "The patient summary must be at an 8th-grade reading level, empathetic, and under 120 words. "
        "The clinician summary can be technical (around 150 words) and mention variant-level signals. "
        "Here is the structured context: "
        f"```json\n{json.dumps(context)}\n```"
    )

    # Try several common SDK methods and gracefully fall back.
    data = None
    def _trim_text(s: str, max_words: int) -> str:
        words = s.strip().split()
        if len(words) <= max_words:
            return s.strip()
        return " ".join(words[:max_words]) + "..."

    try:
        # Primary: try a per-model client generate (if created earlier)
        if gemini_client and hasattr(gemini_client, "generate_content"):
            resp = gemini_client.generate_content(prompt)
            if hasattr(resp, "text"):
                data = json.loads(resp.text)
            elif hasattr(resp, "content"):
                data = json.loads(resp.content)
            elif isinstance(resp, dict):
                data = resp
    except Exception as exc:
        logger.debug("gemini_client.generate_content failed: %s", exc)

    if data is None:
        try:
            # SDK-level generate_text fallback
            if hasattr(genai, "generate_text"):
                resp = genai.generate_text(model=GEMINI_MODEL, prompt=prompt)
                text = getattr(resp, "text", None) or getattr(resp, "content", None) or str(resp)
                try:
                    data = json.loads(text)
                except Exception:
                    # not JSON, try to do heuristics
                    data = {"patient_summary": text, "clinician_summary": text}
            elif hasattr(genai, "generate"):
                # Some versions call genai.generate
                resp = genai.generate(model=GEMINI_MODEL, prompt=prompt)
                text = getattr(resp, "text", None) or getattr(resp, "content", None) or str(resp)
                try:
                    data = json.loads(text)
                except Exception:
                    data = {"patient_summary": text, "clinician_summary": text}
        except Exception as exc:
            logger.warning("Gemini call fallback failed: %s", exc)
        patient = None
        clinician = None
        if isinstance(data, dict):
            patient = data.get("patient_summary")
            clinician = data.get("clinician_summary") or data.get("clinician")
        # If we have a patient summary but it's too long or missing - fall back and trim
        if not patient:
            patient = format_patient_summary(context["risk_category"], context["risk_score"], context.get("top_features", []))
        patient = _trim_text(patient, 60)  # ~60 words
        if not clinician or not any(k in clinician.lower() for k in ["what:", "why:", "how:"]):
            clinician = format_clinician_summary(context.get("top_features", []), context["risk_category"], context["risk_score"], context.get("genetic_features", {}))
        clinician = _trim_text(clinician, 85)  # ~85 words
        return {"patient": patient, "clinician": clinician}
    # If we reach here with no data, return default summaries.
    return {
        "patient": _default_patient_summary(context["risk_category"], context["risk_score"]),
        "clinician": _default_clinician_summary(context["top_features"]),
    }


@app.post("/predict", response_model=GeneticPredictionResponse)
def predict_genetic_risk(request: GeneticPredictRequest) -> GeneticPredictionResponse:
    """Predict hearing deficiency risk based purely on genetic features."""
    features = request.dict()
    risk_score = calculate_genetic_risk(features)
    
    if risk_score < 0.30:
        risk_category = "Low"
        prediction = "Low risk of hearing impairment"
    elif risk_score < 0.50:
        risk_category = "Moderate"
        prediction = "Moderate risk - genetic counseling recommended"
    elif risk_score < 0.70:
        risk_category = "High"
        prediction = "High risk - further genetic testing recommended"
    else:
        risk_category = "Very High"
        prediction = "Very high risk - comprehensive evaluation needed"
    
    confidence = 1.0 - (4 * risk_score * (1 - risk_score))
    confidence = max(0.5, min(confidence, 0.95))

    top_features = get_top_contributing_features(features)
    ai_context = {
        "sample_id": request.sample_id,
        "risk_score": round(risk_score, 4),
        "risk_category": risk_category,
        "prediction": prediction,
        "top_features": top_features,
        "confidence": round(confidence, 4),
        "genetic_features": features,
    }
    ai_insights = generate_ai_insights(ai_context)
    
    return GeneticPredictionResponse(
        sample_id=features.get('sample_id'),
        genetic_risk_score=round(risk_score, 4),
        risk_category=risk_category,
        confidence=round(confidence, 4),
        prediction=prediction,
        top_features=top_features,
        patient_explanation=ai_insights["patient"],
        clinician_explanation=ai_insights["clinician"],
    )


@app.get("/health")
def health_check() -> Dict[str, Any]:
    return {
        "status": "ok",
        "message": "Genetic hearing risk prediction service ready.",
        "version": "2.0.0",
        "approach": "genetics-only",
        "gemini_enabled": GEMINI_ENABLED,
        "gemini_model": GEMINI_MODEL if GEMINI_ENABLED else None,
    }


@app.get("/features")
def list_features() -> Dict[str, Any]:
    """List all genetic features used for prediction"""
    return {
        "genetic_profile": ["ethnicity", "family_history_hearing_loss", "consanguinity", 
                           "syndromic_genetic_condition", "mtdna_variant_detected",
                           "polygenic_risk_score", "gjb2_carrier", "num_affected_relatives"],
        "variant_counts": ["total_variant_count", "pathogenic_variant_count",
                          "likely_pathogenic_count", "vus_count", "benign_variant_count"],
        "gene_specific": ["has_gjb2_variant", "has_slc26a4_variant", "has_otof_variant",
                         "has_myo7a_variant", "has_cdh23_variant", "has_tmc1_variant"],
        "zygosity_impact": ["homozygous_variant_count", "compound_het_count",
                           "high_impact_count", "moderate_impact_count"],
        "scoring": ["max_cadd_score", "mean_cadd_score", "max_revel_score",
                   "rare_variant_count", "unique_genes_affected"]
    }


@app.get("/genes")
def list_hearing_genes() -> Dict[str, Any]:
    """List hearing loss genes analyzed"""
    return {
        "GJB2": {"name": "Connexin 26", "inheritance": "AR", "prevalence": "50%"},
        "SLC26A4": {"name": "Pendrin", "inheritance": "AR", "prevalence": "10%"},
        "OTOF": {"name": "Otoferlin", "inheritance": "AR", "prevalence": "2-3%"},
        "MYO7A": {"name": "Myosin VIIA", "inheritance": "AR/AD", "prevalence": "3-5%"},
        "CDH23": {"name": "Cadherin 23", "inheritance": "AR", "prevalence": "2-3%"},
        "TMC1": {"name": "TMC1", "inheritance": "AR/AD", "prevalence": "1-2%"},
    }


@app.get("/models")
def list_models() -> Dict[str, Any]:
    """Return loaded model information for the API"""
    return {
        "loaded_models": [],
        "llm_insights": {"enabled": GEMINI_ENABLED, "model": GEMINI_MODEL if GEMINI_ENABLED else None},
    }
