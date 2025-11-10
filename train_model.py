import json
import os
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier

DATA_DIR = Path(__file__).resolve().parent / "data"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

LABEL_MAPPING = {
    "Low": [0.0, 0.25],
    "Moderate": [0.25, 0.6],
    "High": [0.6, 1.0],
}

NUMERIC_FEATURES: List[str] = [
    "birth_gestation_weeks",
    "premature",
    "birth_weight_g",
    "apgar_1min",
    "apgar_5min",
    "nicu_days",
    "ventilator",
    "maternal_ototoxic_meds",
    "maternal_diabetes",
    "maternal_hypertension",
    "alcohol_or_drug_exposure",
    "family_history_hearing_loss",
    "bilirubin_mg_dL",
    "phototherapy",
    "exchange_transfusion",
    "sepsis_or_meningitis",
    "ear_anatomy_abnormality",
]

CATEGORICAL_FEATURES: List[str] = [
    "sex",
    "maternal_infection",
    "genetic_condition",
    "oae_result",
    "aabr_result",
]

DROP_COLUMNS = [
    "patient_id",
    "first_name",
    "last_name",
    "synthetic_ground_truth_prob",
    "label_risk",
]

TARGET_NAME = "hearing_deficiency_outcome"


def load_dataset() -> pd.DataFrame:
    dataset_path = DATA_DIR / "dataset_full.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(
            "Dataset not found. Please run generate_dataset.py before training."
        )
    df = pd.read_csv(dataset_path)
    return df


def make_features_and_target(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    working_df = df.drop(columns=DROP_COLUMNS, errors="ignore").copy()
    y_prob = df["synthetic_ground_truth_prob"].values
    rng = np.random.default_rng(RANDOM_STATE)
    y = rng.binomial(1, y_prob)
    working_df[TARGET_NAME] = y

    X = working_df.drop(columns=[TARGET_NAME])
    y_series = working_df[TARGET_NAME]
    return {"X": X, "y": y_series, "y_prob": y_prob}


def build_pipeline() -> Pipeline:
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    base_model = XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=os.cpu_count() or 1,
    )

    calibrated_model = CalibratedClassifierCV(
        base_estimator=base_model,
        method="sigmoid",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", calibrated_model),
        ]
    )
    return pipeline


def evaluate_model(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    proba = pipeline.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, proba)
    brier = brier_score_loss(y_test, proba)
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds).tolist()
    class_report = classification_report(y_test, preds, output_dict=True)

    metrics = {
        "roc_auc": float(auc),
        "brier_score": float(brier),
        "accuracy": float(acc),
        "confusion_matrix": cm,
        "classification_report": class_report,
    }
    return metrics


def save_artifacts(
    pipeline: Pipeline,
    scaler: StandardScaler,
    feature_names: List[str],
    metrics: Dict[str, float],
    X_train_idx: List[int],
    X_test_idx: List[int],
) -> None:
    joblib.dump(pipeline, ARTIFACTS_DIR / "model.joblib")
    joblib.dump(scaler, ARTIFACTS_DIR / "scaler.joblib")
    (ARTIFACTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    schema = {
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "all_features": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "encoded_feature_names": feature_names,
        "label_mapping": LABEL_MAPPING,
        "target_name": TARGET_NAME,
    }
    (ARTIFACTS_DIR / "columns.json").write_text(json.dumps(schema, indent=2))

    split_indices = {
        "train_indices": X_train_idx,
        "test_indices": X_test_idx,
    }
    (ARTIFACTS_DIR / "train_test_split_indices.json").write_text(
        json.dumps(split_indices, indent=2)
    )



def main() -> None:
    df = load_dataset()
    prepared = make_features_and_target(df)
    X = prepared["X"]
    y = prepared["y"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    metrics = evaluate_model(pipeline, X_test, y_test)

    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    scaler: StandardScaler = preprocessor.named_transformers_["num"].named_steps["scaler"]
    feature_names = preprocessor.get_feature_names_out().tolist()

    save_artifacts(
        pipeline,
        scaler,
        feature_names,
        metrics,
        X_train.index.tolist(),
        X_test.index.tolist(),
    )

    print("Training complete. Metrics:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
