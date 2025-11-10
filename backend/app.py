import os
from typing import Any, Dict, List, Optional, Literal

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

from . import models_utils

app = FastAPI(
    title="Synthetic Newborn Hearing Risk API",
    description=(
        "Prototype API powered by a synthetic dataset predicting the probability "
        "of neonatal hearing deficiency. This system is for demonstration only and "
        "must not be used for medical decision-making."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_patients_dataframe() -> pd.DataFrame:
    df = models_utils.load_dataset()
    return df


PATIENTS_DF = load_patients_dataframe()
DATASET_COLUMNS = PATIENTS_DF.columns.tolist()
SCHEMA = models_utils.load_schema()
MODEL_FEATURES = SCHEMA["all_features"]
LABEL_MAPPING = SCHEMA["label_mapping"]


class PatientBase(BaseModel):
    first_name: str = Field(..., min_length=1)
    last_name: str = Field(..., min_length=1)
    sex: Literal['M', 'F']
    birth_gestation_weeks: int = Field(..., ge=24, le=42)
    premature: Optional[int] = Field(None, ge=0, le=1)
    birth_weight_g: int = Field(..., ge=450, le=4500)
    apgar_1min: int = Field(..., ge=0, le=10)
    apgar_5min: int = Field(..., ge=0, le=10)
    nicu_days: int = Field(..., ge=0, le=60)
    ventilator: int = Field(..., ge=0, le=1)
    maternal_infection: Literal['none', 'CMV', 'Rubella', 'Toxoplasmosis', 'Syphilis', 'Zika']
    maternal_ototoxic_meds: int = Field(..., ge=0, le=1)
    maternal_diabetes: int = Field(..., ge=0, le=1)
    maternal_hypertension: int = Field(..., ge=0, le=1)
    alcohol_or_drug_exposure: int = Field(..., ge=0, le=1)
    family_history_hearing_loss: int = Field(..., ge=0, le=1)
    genetic_condition: Literal['none', 'GJB2', 'Pendred', 'Usher', 'Other']
    bilirubin_mg_dL: float = Field(..., ge=0.0, le=40.0)
    phototherapy: int = Field(..., ge=0, le=1)
    exchange_transfusion: int = Field(..., ge=0, le=1)
    sepsis_or_meningitis: int = Field(..., ge=0, le=1)
    ear_anatomy_abnormality: int = Field(..., ge=0, le=1)
    oae_result: Literal['pass', 'refer']
    aabr_result: Literal['pass', 'refer']
    consent_for_research: int = Field(..., ge=0, le=1)

    @validator("premature", always=True)
    def compute_prematurity(cls, v: Optional[int], values: Dict[str, Any]) -> int:
        if v is None:
            weeks = values.get("birth_gestation_weeks")
            if weeks is None:
                return 0
            return 1 if weeks < 37 else 0
        return v

    class Config:
        orm_mode = True


class PatientPredictRequest(PatientBase):
    patient_id: Optional[str] = None


class PatientAddRequest(PatientPredictRequest):
    pass


class PatientPredictionResponse(BaseModel):
    patient_id: Optional[str]
    probability: float
    risk_category: str
    confidence: float
    explanation: Dict[str, Any]
    disclaimer: str = Field(
        "Synthetic model output for prototyping only. Not a medical diagnostic tool.",
        const=True,
    )


class PatientRecordResponse(BaseModel):
    record: Dict[str, Any]
    prediction: PatientPredictionResponse


class PatientRecordListResponse(BaseModel):
    matches: List[PatientRecordResponse]


def to_model_features(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {feature: payload[feature] for feature in MODEL_FEATURES}


def build_patient_row(payload: Dict[str, Any], prediction: models_utils.PredictionResult) -> Dict[str, Any]:
    row = {col: payload.get(col) for col in DATASET_COLUMNS if col in payload}
    row["synthetic_ground_truth_prob"] = prediction.probability
    row["label_risk"] = prediction.risk_category
    return row


@app.post("/predict", response_model=PatientPredictionResponse)
def predict(request: PatientPredictRequest) -> PatientPredictionResponse:
    payload = request.dict()
    model_features = to_model_features(payload)
    prediction = models_utils.predict_probability(model_features)

    return PatientPredictionResponse(
        patient_id=payload.get("patient_id"),
        probability=prediction.probability,
        risk_category=prediction.risk_category,
        confidence=prediction.confidence,
        explanation=prediction.explanation,
    )


@app.get("/patient/{patient_name}", response_model=PatientRecordListResponse)
def get_patient(patient_name: str) -> PatientRecordListResponse:
    if not patient_name:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Name required")

    name_query = patient_name.strip().lower()
    full_name_series = (
        PATIENTS_DF["first_name"].str.lower() + " " + PATIENTS_DF["last_name"].str.lower()
    )
    mask = full_name_series == name_query
    if not mask.any():
        mask = PATIENTS_DF["first_name"].str.lower().str.contains(name_query)
    matches_df = PATIENTS_DF[mask]

    if matches_df.empty:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found")

    results: List[PatientRecordResponse] = []
    for _, row in matches_df.iterrows():
        payload = row.to_dict()
        model_features = to_model_features(payload)
        prediction = models_utils.predict_probability(model_features)
        record = row.to_dict()
        record.update(
            {
                "model_probability": prediction.probability,
                "model_risk_category": prediction.risk_category,
                "model_confidence": prediction.confidence,
            }
        )
        results.append(
            PatientRecordResponse(
                record=record,
                prediction=PatientPredictionResponse(
                    patient_id=row.get("patient_id"),
                    probability=prediction.probability,
                    risk_category=prediction.risk_category,
                    confidence=prediction.confidence,
                    explanation=prediction.explanation,
                ),
            )
        )

    return PatientRecordListResponse(matches=results)


@app.post("/add_patient", response_model=PatientRecordResponse)
def add_patient(request: PatientAddRequest) -> PatientRecordResponse:
    global PATIENTS_DF

    payload = request.dict()
    if payload["consent_for_research"] not in (0, 1):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid consent flag")

    patient_id = payload.get("patient_id")
    if not patient_id:
        patient_id = models_utils.generate_next_patient_id(PATIENTS_DF["patient_id"].tolist())
        payload["patient_id"] = patient_id

    model_features = to_model_features(payload)
    prediction = models_utils.predict_probability(model_features)

    row = build_patient_row(payload, prediction)

    if payload["consent_for_research"] == 1:
        models_utils.append_new_entry(row)
        new_row = {**row}
        for col in DATASET_COLUMNS:
            if col not in new_row:
                new_row[col] = None
        new_row_df = pd.DataFrame([new_row])
        PATIENTS_DF = pd.concat([PATIENTS_DF, new_row_df], ignore_index=True)

    return PatientRecordResponse(
        record=row,
        prediction=PatientPredictionResponse(
            patient_id=patient_id,
            probability=prediction.probability,
            risk_category=prediction.risk_category,
            confidence=prediction.confidence,
            explanation=prediction.explanation,
        ),
    )


class RetrainRequest(BaseModel):
    secret_key: Optional[str] = None


@app.post("/retrain")
def retrain_endpoint(request: RetrainRequest) -> Dict[str, str]:
    enable_flag = os.getenv("ENABLE_RETRAIN", "false").lower() == "true"
    if not enable_flag:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Retraining is disabled in this deployment. Set ENABLE_RETRAIN=true to enable.",
        )

    expected_key = os.getenv("RETRAIN_SECRET")
    if expected_key and request.secret_key != expected_key:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid secret key")

    # Placeholder for retraining logic: would combine new entries and original data.
    # Intentionally not triggering automatically for safety.
    return {
        "status": "accepted",
        "message": "Retraining is enabled but not automatically triggered. Run train_model.py manually.",
    }


@app.get("/health")
def health_check() -> Dict[str, str]:
    return {
        "status": "ok",
        "message": "Synthetic newborn hearing risk service ready.",
        "disclaimer": "Synthetic prototype only; not for medical use.",
    }
