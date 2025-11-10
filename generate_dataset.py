import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from faker import Faker

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

NUM_PATIENTS = 50_000
SEED = 42

faker = Faker()
faker.seed_instance(SEED)
np.random.seed(SEED)


PATIENT_SCHEMA_COLUMNS: List[str] = [
    "patient_id",
    "first_name",
    "last_name",
    "sex",
    "birth_gestation_weeks",
    "premature",
    "birth_weight_g",
    "apgar_1min",
    "apgar_5min",
    "nicu_days",
    "ventilator",
    "maternal_infection",
    "maternal_ototoxic_meds",
    "maternal_diabetes",
    "maternal_hypertension",
    "alcohol_or_drug_exposure",
    "family_history_hearing_loss",
    "genetic_condition",
    "bilirubin_mg_dL",
    "phototherapy",
    "exchange_transfusion",
    "sepsis_or_meningitis",
    "ear_anatomy_abnormality",
    "oae_result",
    "aabr_result",
    "consent_for_research",
    "synthetic_ground_truth_prob",
    "label_risk",
]


def generate_unique_names(n: int) -> Tuple[List[str], List[str]]:
    first_names: List[str] = []
    last_names: List[str] = []
    seen_names = set()

    while len(first_names) < n:
        first = faker.first_name()
        last = faker.last_name()
        key = (first, last)
        if key in seen_names:
            continue
        seen_names.add(key)
        first_names.append(first)
        last_names.append(last)
    return first_names, last_names


def generate_patient_ids(n: int) -> List[str]:
    return [f"NB-{idx:06d}" for idx in range(1, n + 1)]


def sample_maternal_infection(size: int) -> List[str]:
    options = ["none", "CMV", "Rubella", "Toxoplasmosis", "Syphilis", "Zika"]
    probs = [0.95, 0.012, 0.01, 0.01, 0.012, 0.006]
    return list(np.random.choice(options, size=size, p=probs))


def sample_genetic_condition(size: int) -> List[str]:
    options = ["none", "GJB2", "Pendred", "Usher", "Other"]
    probs = [0.982, 0.007, 0.004, 0.003, 0.004]
    return list(np.random.choice(options, size=size, p=probs))


def base_risk_logit(df: pd.DataFrame) -> np.ndarray:
    intercept = -3.15
    gestation_contrib = -0.06 * np.clip(40 - df["birth_gestation_weeks"], 0, None)
    weight_contrib = -0.0007 * np.clip(3000 - df["birth_weight_g"], 0, None)
    apgar_contrib = -0.18 * np.clip(8 - df["apgar_5min"], 0, None)
    nicu_contrib = 0.03 * np.clip(df["nicu_days"], 0, 60)
    ventilator_contrib = 0.6 * df["ventilator"]

    infection_weights = {
        "none": 0.0,
        "CMV": 1.2,
        "Rubella": 1.0,
        "Toxoplasmosis": 0.8,
        "Syphilis": 0.7,
        "Zika": 1.0,
    }
    infection_contrib = df["maternal_infection"].map(infection_weights)

    ototoxic_contrib = 0.9 * df["maternal_ototoxic_meds"]
    diabetes_contrib = 0.2 * df["maternal_diabetes"]
    hypertension_contrib = 0.15 * df["maternal_hypertension"]
    exposure_contrib = 0.6 * df["alcohol_or_drug_exposure"]
    family_history_contrib = 0.9 * df["family_history_hearing_loss"]

    genetic_weights = {
        "none": 0.0,
        "GJB2": 1.4,
        "Pendred": 1.2,
        "Usher": 1.3,
        "Other": 0.9,
    }
    genetic_contrib = df["genetic_condition"].map(genetic_weights)

    bilirubin_contrib = 0.08 * np.clip(df["bilirubin_mg_dL"] - 10.0, 0, None)
    phototherapy_contrib = 0.3 * df["phototherapy"]
    exchange_contrib = 0.9 * df["exchange_transfusion"]
    sepsis_contrib = 1.0 * df["sepsis_or_meningitis"]
    ear_contrib = 0.8 * df["ear_anatomy_abnormality"]

    oae_contrib = 0.9 * (df["oae_result"] == "refer").astype(float)
    aabr_contrib = 1.0 * (df["aabr_result"] == "refer").astype(float)

    logit = (
        intercept
        + gestation_contrib
        + weight_contrib
        + apgar_contrib
        + nicu_contrib
        + ventilator_contrib
        + infection_contrib
        + ototoxic_contrib
        + diabetes_contrib
        + hypertension_contrib
        + exposure_contrib
        + family_history_contrib
        + genetic_contrib
        + bilirubin_contrib
        + phototherapy_contrib
        + exchange_contrib
        + sepsis_contrib
        + ear_contrib
        + oae_contrib
        + aabr_contrib
    )
    return logit


def logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def assign_risk_category(prob: float) -> str:
    if prob <= 0.25:
        return "Low"
    if prob <= 0.60:
        return "Moderate"
    return "High"


def main() -> None:
    ids = generate_patient_ids(NUM_PATIENTS)
    first_names, last_names = generate_unique_names(NUM_PATIENTS)

    sex = np.random.choice(["M", "F"], size=NUM_PATIENTS)

    gestation = np.clip(
        np.round(np.random.normal(loc=39, scale=1.8, size=NUM_PATIENTS)).astype(int),
        24,
        42,
    )
    premature = (gestation < 37).astype(int)

    birth_weight = (
        2600 + 120 * (gestation - 30) + np.random.normal(0, 250, size=NUM_PATIENTS)
    )
    birth_weight = np.clip(birth_weight, 450, 4500).astype(int)

    apgar_5min = np.clip(
        np.round(np.random.normal(loc=8.5, scale=1.2, size=NUM_PATIENTS)), 0, 10
    ).astype(int)
    apgar_1min = np.clip(
        apgar_5min - np.random.choice([0, 1, 2], p=[0.5, 0.35, 0.15], size=NUM_PATIENTS),
        0,
        10,
    )

    maternal_infection = sample_maternal_infection(NUM_PATIENTS)
    maternal_ototoxic = np.random.binomial(1, 0.015, size=NUM_PATIENTS)
    maternal_diabetes = np.random.binomial(1, 0.12, size=NUM_PATIENTS)
    maternal_hypertension = np.random.binomial(1, 0.18, size=NUM_PATIENTS)
    alcohol_drug = np.random.binomial(1, 0.06, size=NUM_PATIENTS)
    family_history = np.random.binomial(1, 0.05, size=NUM_PATIENTS)
    genetic_condition = sample_genetic_condition(NUM_PATIENTS)

    base_bilirubin = np.random.gamma(shape=2.5, scale=3.0, size=NUM_PATIENTS)
    bilirubin = base_bilirubin
    bilirubin += premature * np.random.uniform(0, 4, size=NUM_PATIENTS)
    bilirubin += np.random.binomial(1, 0.05, size=NUM_PATIENTS) * np.random.uniform(2, 6, size=NUM_PATIENTS)
    bilirubin = np.clip(bilirubin, 0.0, 40.0)

    phototherapy = (bilirubin > 15).astype(int)
    exchange_transfusion = ((bilirubin > 25) & (np.random.random(NUM_PATIENTS) < 0.3)).astype(int)

    sepsis = np.random.binomial(1, 0.01, size=NUM_PATIENTS)
    ear_abnormality = np.random.binomial(1, 0.02, size=NUM_PATIENTS)

    ventilator = np.random.binomial(1, 0.05 + premature * 0.1)

    nicu_base = np.random.poisson(lam=1.2, size=NUM_PATIENTS)
    nicu_additional = (
        (premature * np.random.poisson(5, size=NUM_PATIENTS))
        + (ventilator * np.random.poisson(6, size=NUM_PATIENTS))
        + (sepsis * np.random.poisson(7, size=NUM_PATIENTS))
        + ((apgar_1min <= 4).astype(int) * np.random.poisson(4, size=NUM_PATIENTS))
    )
    nicu_days = np.clip(nicu_base + nicu_additional, 0, 60)

    sepsis = np.where(nicu_days > 5, np.maximum(sepsis, np.random.binomial(1, 0.03, size=NUM_PATIENTS)), sepsis)
    bilirubin += sepsis * np.random.uniform(1.5, 5.0, size=NUM_PATIENTS)
    bilirubin = np.clip(bilirubin, 0.0, 40.0)
    phototherapy = np.where(bilirubin > 15, 1, phototherapy)
    exchange_transfusion = np.where(
        bilirubin > 25,
        np.maximum(exchange_transfusion, (np.random.random(NUM_PATIENTS) < 0.4).astype(int)),
        exchange_transfusion,
    )

    consent = np.random.binomial(1, 0.7, size=NUM_PATIENTS)

    df = pd.DataFrame(
        {
            "patient_id": ids,
            "first_name": first_names,
            "last_name": last_names,
            "sex": sex,
            "birth_gestation_weeks": gestation,
            "premature": premature,
            "birth_weight_g": birth_weight,
            "apgar_1min": apgar_1min,
            "apgar_5min": apgar_5min,
            "nicu_days": nicu_days,
            "ventilator": ventilator,
            "maternal_infection": maternal_infection,
            "maternal_ototoxic_meds": maternal_ototoxic,
            "maternal_diabetes": maternal_diabetes,
            "maternal_hypertension": maternal_hypertension,
            "alcohol_or_drug_exposure": alcohol_drug,
            "family_history_hearing_loss": family_history,
            "genetic_condition": genetic_condition,
            "bilirubin_mg_dL": np.round(bilirubin, 2),
            "phototherapy": phototherapy,
            "exchange_transfusion": exchange_transfusion,
            "sepsis_or_meningitis": sepsis,
            "ear_anatomy_abnormality": ear_abnormality,
            "consent_for_research": consent,
        }
    )

    # Base risk logit without OAE/AABR to simulate screening results
    temp_df = df.copy()
    temp_df["oae_result"] = "pass"
    temp_df["aabr_result"] = "pass"
    base_logit = base_risk_logit(temp_df)
    base_prob = logistic(base_logit)

    refer_prob = np.clip(base_prob * 1.5, 0, 0.9)
    oae_random = np.random.random(NUM_PATIENTS)
    aabr_random = np.random.random(NUM_PATIENTS)
    oae_result = np.where(oae_random < refer_prob, "refer", "pass")
    aabr_result = np.where(aabr_random < refer_prob * 0.8, "refer", "pass")

    df["oae_result"] = oae_result
    df["aabr_result"] = aabr_result

    final_logit = base_risk_logit(df)
    prob = logistic(final_logit)
    prob += np.random.normal(0, 0.02, size=NUM_PATIENTS)
    prob = np.clip(prob, 0, 1)

    df["synthetic_ground_truth_prob"] = np.round(prob, 4)
    df["label_risk"] = df["synthetic_ground_truth_prob"].apply(assign_risk_category)

    df = df[PATIENT_SCHEMA_COLUMNS]

    output_path = DATA_DIR / "dataset_full.csv"
    df.to_csv(output_path, index=False)

    summary = {
        "num_rows": len(df),
        "risk_category_counts": df["label_risk"].value_counts().to_dict(),
        "mean_probability": float(df["synthetic_ground_truth_prob"].mean()),
        "std_probability": float(df["synthetic_ground_truth_prob"].std()),
    }

    (DATA_DIR / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Generated dataset with {len(df)} rows at {output_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
