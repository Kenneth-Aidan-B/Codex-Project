import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATA_DIR = BASE_DIR / "data"

MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
SCALER_PATH = ARTIFACTS_DIR / "scaler.joblib"
SCHEMA_PATH = ARTIFACTS_DIR / "columns.json"


class PredictionResult(BaseModel):
    probability: float
    risk_category: str
    confidence: float
    explanation: Dict[str, Any]


@lru_cache(maxsize=1)
def load_pipeline() -> Pipeline:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model artifact not found. Train the model before starting the API."
        )
    pipeline: Pipeline = joblib.load(MODEL_PATH)
    return pipeline


@lru_cache(maxsize=1)
def load_scaler() -> StandardScaler:
    if not SCALER_PATH.exists():
        raise FileNotFoundError("Scaler artifact not found. Run train_model.py first.")
    scaler: StandardScaler = joblib.load(SCALER_PATH)
    return scaler


@lru_cache(maxsize=1)
def load_schema() -> Dict[str, Any]:
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError("columns.json not found in artifacts directory.")
    return json.loads(SCHEMA_PATH.read_text())


def risk_category_from_probability(prob: float, mapping: Dict[str, List[float]]) -> str:
    for label, (lower, upper) in mapping.items():
        if lower <= prob <= upper:
            return label
    return "High" if prob > 0.6 else "Moderate"


def compute_confidence(prob: float) -> float:
    # A simple heuristic: peaked probabilities yield higher confidence
    confidence = 1.0 - (4 * prob * (1 - prob))
    return float(np.clip(confidence, 0.0, 1.0))


def prepare_input_dataframe(payload: Dict[str, Any], feature_order: List[str]) -> pd.DataFrame:
    df = pd.DataFrame([payload])
    return df[feature_order]


def rule_based_contributions(features: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Mirror the generation weights for a transparent explanation
    weeks = features["birth_gestation_weeks"]
    weight_g = features["birth_weight_g"]
    apgar5 = features["apgar_5min"]
    nicu = features["nicu_days"]

    contributions: List[Dict[str, Any]] = []

    gestation_delta = max(0, 40 - weeks)
    contributions.append(
        {
            "name": "Low gestational age",
            "value": weeks,
            "contribution": round(-0.06 * gestation_delta, 3),
        }
    )
    weight_delta = max(0, 3000 - weight_g)
    contributions.append(
        {
            "name": "Low birth weight",
            "value": weight_g,
            "contribution": round(-0.0007 * weight_delta, 3),
        }
    )
    apgar_delta = max(0, 8 - apgar5)
    contributions.append(
        {
            "name": "APGAR 5min",
            "value": apgar5,
            "contribution": round(-0.18 * apgar_delta, 3),
        }
    )
    contributions.append(
        {
            "name": "NICU days",
            "value": nicu,
            "contribution": round(0.03 * min(nicu, 60), 3),
        }
    )

    weighted_flags = {
        "ventilator": 0.6,
        "maternal_ototoxic_meds": 0.9,
        "maternal_diabetes": 0.2,
        "maternal_hypertension": 0.15,
        "alcohol_or_drug_exposure": 0.6,
        "family_history_hearing_loss": 0.9,
        "phototherapy": 0.3,
        "exchange_transfusion": 0.9,
        "sepsis_or_meningitis": 1.0,
        "ear_anatomy_abnormality": 0.8,
    }
    for key, weight in weighted_flags.items():
        contributions.append(
            {
                "name": key,
                "value": features.get(key, 0),
                "contribution": round(weight * features.get(key, 0), 3),
            }
        )

    infection_weights = {
        "none": 0.0,
        "CMV": 1.2,
        "Rubella": 1.0,
        "Toxoplasmosis": 0.8,
        "Syphilis": 0.7,
        "Zika": 1.0,
    }
    contributions.append(
        {
            "name": "maternal_infection",
            "value": features["maternal_infection"],
            "contribution": infection_weights.get(features["maternal_infection"], 0.0),
        }
    )

    genetic_weights = {
        "none": 0.0,
        "GJB2": 1.4,
        "Pendred": 1.2,
        "Usher": 1.3,
        "Other": 0.9,
    }
    contributions.append(
        {
            "name": "genetic_condition",
            "value": features["genetic_condition"],
            "contribution": genetic_weights.get(features["genetic_condition"], 0.0),
        }
    )

    bilirubin_delta = max(0.0, features["bilirubin_mg_dL"] - 10.0)
    contributions.append(
        {
            "name": "bilirubin_mg_dL",
            "value": features["bilirubin_mg_dL"],
            "contribution": round(0.08 * bilirubin_delta, 3),
        }
    )

    oae_contrib = 0.9 if features["oae_result"] == "refer" else 0.0
    aabr_contrib = 1.0 if features["aabr_result"] == "refer" else 0.0
    contributions.append(
        {
            "name": "oae_result",
            "value": features["oae_result"],
            "contribution": oae_contrib,
        }
    )
    contributions.append(
        {
            "name": "aabr_result",
            "value": features["aabr_result"],
            "contribution": aabr_contrib,
        }
    )

    contributions_sorted = sorted(
        contributions,
        key=lambda item: abs(item["contribution"]),
        reverse=True,
    )
    return contributions_sorted[:5]


def predict_probability(features: Dict[str, Any]) -> PredictionResult:
    schema = load_schema()
    feature_order = schema["all_features"]
    label_mapping = schema["label_mapping"]

    pipeline = load_pipeline()
    df = prepare_input_dataframe(features, feature_order)
    probability = float(pipeline.predict_proba(df)[0, 1])
    risk_category = risk_category_from_probability(probability, label_mapping)
    confidence = compute_confidence(probability)

    explanation = {"top_features": rule_based_contributions(features)}

    return PredictionResult(
        probability=probability,
        risk_category=risk_category,
        confidence=confidence,
        explanation=explanation,
    )


def load_dataset() -> pd.DataFrame:
    dataset_path = DATA_DIR / "dataset_full.csv"
    if not dataset_path.exists():
        raise FileNotFoundError("Dataset not found. Run generate_dataset.py first.")
    df = pd.read_csv(dataset_path)
    return df


@lru_cache(maxsize=1)
def dataset_columns() -> List[str]:
    return load_dataset().columns.tolist()


def append_new_entry(row: Dict[str, Any]) -> None:
    new_entries_path = DATA_DIR / "new_entries.csv"
    columns = dataset_columns()
    df = pd.DataFrame([row], columns=columns)
    header = not new_entries_path.exists()
    df.to_csv(new_entries_path, mode="a", header=header, index=False)


def generate_next_patient_id(existing_ids: List[str]) -> str:
    prefix = "NB-"
    numbers = [int(pid.split("-")[1]) for pid in existing_ids if pid.startswith(prefix)]
    next_number = max(numbers, default=0) + 1
    return f"{prefix}{next_number:06d}"
