#!/usr/bin/env python3
"""
FastAPI application for hearing deficiency prediction and explanation.
Provides /predict and /explain/{sample_id} endpoints.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"
EXPLANATIONS_DIR = RESULTS_DIR / "explanations"

# Initialize FastAPI
app = FastAPI(
    title="Hearing Deficiency Prediction API",
    description="ML-powered API for hearing deficiency risk prediction and explanation",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model cache
MODELS = {}
PREPROCESSING_ARTIFACTS = None
FEATURE_NAMES = None


class PredictionInput(BaseModel):
    """Input schema for single prediction"""
    age_months: int
    sex: str
    ethnicity: str
    birth_weight_g: int
    gestational_age_weeks: float
    premature: int
    apgar_1min: int
    apgar_5min: int
    nicu_days: int
    mechanical_ventilation_days: int
    hyperbilirubinemia: int
    bilirubin_max_mg_dl: float
    ototoxic_medications: int
    aminoglycoside_days: int
    loop_diuretic_days: int
    maternal_cmv_infection: int
    maternal_rubella: int
    maternal_toxoplasmosis: int
    family_history_hearing_loss: int
    consanguinity: int
    syndromic_features: int
    craniofacial_anomalies: int
    oae_result: str
    aabr_result: str
    # Genomic features (optional, default to 0)
    pathogenic_variant_count: int = 0
    has_gjb2_variant: int = 0
    has_slc26a4_variant: int = 0
    has_otof_variant: int = 0
    max_cadd_score: float = 0.0
    homozygous_variant_count: int = 0
    compound_het_count: int = 0
    high_impact_variant_count: int = 0


class PredictionOutput(BaseModel):
    """Output schema for predictions"""
    sample_id: str
    risk_score: float
    prediction: str
    confidence: float
    top_features: List[Dict[str, float]]
    model_used: str


def load_models():
    """Load all trained models"""
    global MODELS
    
    model_names = ['RandomForest', 'XGBoost', 'SVM']
    
    for model_name in model_names:
        model_path = MODELS_DIR / model_name / "model_latest.joblib"
        if model_path.exists():
            try:
                MODELS[model_name] = joblib.load(model_path)
                logger.info(f"✓ Loaded {model_name}")
            except Exception as e:
                logger.error(f"Error loading {model_name}: {e}")
    
    if not MODELS:
        logger.error("No models loaded!")
    else:
        logger.info(f"Loaded {len(MODELS)} models")


def load_preprocessing_artifacts():
    """Load preprocessing artifacts"""
    global PREPROCESSING_ARTIFACTS, FEATURE_NAMES
    
    artifacts_path = PROCESSED_DIR / "preprocessing_artifacts.joblib"
    metadata_path = PROCESSED_DIR / "metadata.json"
    
    if artifacts_path.exists():
        PREPROCESSING_ARTIFACTS = joblib.load(artifacts_path)
        logger.info("✓ Loaded preprocessing artifacts")
    
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            FEATURE_NAMES = metadata['feature_names']
            logger.info(f"✓ Loaded feature names ({len(FEATURE_NAMES)} features)")


def preprocess_input(input_data: Dict) -> np.ndarray:
    """Preprocess input data for prediction"""
    # Create DataFrame from input
    df = pd.DataFrame([input_data])
    
    # Ensure all expected features are present
    for feature in FEATURE_NAMES:
        if feature not in df.columns:
            df[feature] = 0
    
    # Select features in correct order
    df = df[FEATURE_NAMES]
    
    # Apply preprocessing if available
    if PREPROCESSING_ARTIFACTS:
        # Encode categorical features
        label_encoders = PREPROCESSING_ARTIFACTS['label_encoders']
        for col, le in label_encoders.items():
            if col in df.columns:
                try:
                    df[col] = le.transform(df[col].fillna('unknown'))
                except:
                    df[col] = 0  # Unknown category
        
        # Impute numerical features
        imputer = PREPROCESSING_ARTIFACTS['imputer']
        numerical_cols = PREPROCESSING_ARTIFACTS['numerical_cols']
        if numerical_cols and len(numerical_cols) > 0:
            numerical_cols = [col for col in numerical_cols if col in df.columns]
            if numerical_cols:
                df[numerical_cols] = imputer.transform(df[numerical_cols])
        
        # Scale numerical features
        scaler = PREPROCESSING_ARTIFACTS['scaler']
        if numerical_cols and len(numerical_cols) > 0:
            df[numerical_cols] = scaler.transform(df[numerical_cols])
    
    return df.values


@app.on_event("startup")
async def startup_event():
    """Load models and artifacts on startup"""
    logger.info("Starting API...")
    load_models()
    load_preprocessing_artifacts()
    logger.info("API ready!")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Hearing Deficiency Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Make predictions",
            "/explain/{sample_id}": "GET - Get SHAP explanations",
            "/health": "GET - Health check",
            "/models": "GET - List loaded models"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": len(MODELS),
        "models": list(MODELS.keys())
    }


@app.get("/models")
async def list_models():
    """List loaded models"""
    return {
        "models": list(MODELS.keys()),
        "count": len(MODELS)
    }


@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    """Make a prediction"""
    try:
        # Check if models are loaded
        if not MODELS:
            raise HTTPException(status_code=500, detail="No models loaded")
        
        # Convert input to dict
        data_dict = input_data.dict()
        
        # Preprocess input
        X = preprocess_input(data_dict)
        
        # Use RandomForest as default model
        model_name = "RandomForest" if "RandomForest" in MODELS else list(MODELS.keys())[0]
        model = MODELS[model_name]
        
        # Make prediction
        prediction = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        risk_score = float(proba[1])
        confidence = float(max(proba))
        
        # Determine prediction label
        prediction_label = "High Risk" if prediction == 1 else "Low Risk"
        
        # Get feature importance (if available)
        top_features = []
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            top_indices = np.argsort(importances)[-10:][::-1]
            top_features = [
                {"feature": FEATURE_NAMES[idx], "importance": float(importances[idx])}
                for idx in top_indices
            ]
        
        # Generate sample ID
        sample_id = f"prediction_{hash(str(data_dict))}"[:16]
        
        return PredictionOutput(
            sample_id=sample_id,
            risk_score=risk_score,
            prediction=prediction_label,
            confidence=confidence,
            top_features=top_features,
            model_used=model_name
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/explain/{sample_id}")
async def explain(sample_id: str, model: str = "RandomForest"):
    """Get SHAP explanation for a sample"""
    try:
        # Look for explanation file
        explanation_path = EXPLANATIONS_DIR / f"{sample_id}_{model}.json"
        
        if not explanation_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Explanation not found for sample {sample_id} and model {model}"
            )
        
        with open(explanation_path, 'r') as f:
            explanation = json.load(f)
        
        return explanation
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Explanation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    """Make batch predictions from CSV file"""
    try:
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents))
        
        # Check if models are loaded
        if not MODELS:
            raise HTTPException(status_code=500, detail="No models loaded")
        
        # Use RandomForest as default model
        model_name = "RandomForest" if "RandomForest" in MODELS else list(MODELS.keys())[0]
        model = MODELS[model_name]
        
        predictions = []
        
        for idx, row in df.iterrows():
            # Preprocess
            X = preprocess_input(row.to_dict())
            
            # Predict
            prediction = model.predict(X)[0]
            proba = model.predict_proba(X)[0]
            risk_score = float(proba[1])
            
            predictions.append({
                "row": idx,
                "risk_score": risk_score,
                "prediction": "High Risk" if prediction == 1 else "Low Risk"
            })
        
        return {
            "count": len(predictions),
            "predictions": predictions,
            "model_used": model_name
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
