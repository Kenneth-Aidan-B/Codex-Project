# Synthetic Newborn Hearing Risk Project

> **Disclaimer:** This repository contains a fully synthetic dataset and prototype machine learning pipeline intended only for demonstrations and rapid prototyping. It is **not** a medical device, provides no clinical guidance, and must never be used for patient care. Consult a licensed audiologist or neonatologist for medical decisions.

## Project Overview

This project generates a 50,000-row synthetic dataset of newborn patients with clinically inspired features, trains a calibrated machine learning model to estimate the probability of hearing deficiency, and exposes the results through a FastAPI backend and a React frontend.

Key capabilities:

- Deterministic dataset generation using Faker and NumPy with reproducible seeds.
- Rule-driven synthetic ground-truth probability that incorporates known neonatal hearing risk factors.
- XGBoost classifier wrapped with calibration to produce probability estimates and risk tiers (Low / Moderate / High).
- FastAPI service providing endpoints for predictions, patient lookup, consent-aware patient creation, and a gated retraining hook.
- React single page interface for searching patients, visualising predictions, and submitting new patient values.

## Repository Structure

```
.
├── artifacts/                # Model artifacts (populated after training)
├── backend/                  # FastAPI application
├── data/                     # Generated datasets and new entries
├── frontend/                 # React single page app (create-react-app based)
├── generate_dataset.py       # Script to create the synthetic dataset
├── train_model.py            # Script to train & evaluate the ML model
├── requirements.txt          # Python dependencies
├── run_generate_and_train.sh # Helper script: generate dataset + train
├── run_backend.sh            # Helper script: launch FastAPI with uvicorn
├── run_frontend.sh           # Helper script: launch React dev server
└── README.md
```

## Prerequisites

- Python 3.10+
- Node.js 18+ and npm 9+

## Python Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Generate the Synthetic Dataset

```bash
python generate_dataset.py
```

This will create `data/dataset_full.csv` (50,000 unique rows) and `data/dataset_summary.json`.

## Train the Model

```bash
python train_model.py
```

Artifacts saved to `artifacts/`:

- `model.joblib`: preprocessing + calibrated classifier pipeline
- `scaler.joblib`: fitted StandardScaler for numeric features
- `columns.json`: schema metadata and label mapping
- `metrics.json`: evaluation metrics on the held-out test set
- `train_test_split_indices.json`: indices for reproducibility

## Helper Script

Run generation and training sequentially:

```bash
./run_generate_and_train.sh
```

(Ensure the script is executable: `chmod +x run_generate_and_train.sh`).

## Running the FastAPI Backend

```bash
./run_backend.sh
```

The script expands to:

```bash
uvicorn backend.app:app --reload --port 8000
```

Environment variables:

- `ENABLE_RETRAIN=true` (optional) – unlocks the `/retrain` endpoint.
- `RETRAIN_SECRET=<value>` (optional) – shared secret required when retrain is enabled.

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service heartbeat + disclaimer |
| `/predict` | POST | Accept raw newborn features, return probability, risk tier, confidence, explanation |
| `/patient/{name}` | GET | Exact/partial case-insensitive name lookup returning stored metrics and predictions |
| `/add_patient` | POST | Add a new patient row (consent required to persist), returns saved row + prediction |
| `/retrain` | POST | Gated retrain hook – disabled unless `ENABLE_RETRAIN=true` |

Example `curl` calls:

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "first_name": "Test",
    "last_name": "Infant",
    "sex": "F",
    "birth_gestation_weeks": 36,
    "premature": 1,
    "birth_weight_g": 2500,
    "apgar_1min": 7,
    "apgar_5min": 8,
    "nicu_days": 5,
    "ventilator": 0,
    "maternal_infection": "none",
    "maternal_ototoxic_meds": 0,
    "maternal_diabetes": 0,
    "maternal_hypertension": 0,
    "alcohol_or_drug_exposure": 0,
    "family_history_hearing_loss": 1,
    "genetic_condition": "none",
    "bilirubin_mg_dL": 8.5,
    "phototherapy": 0,
    "exchange_transfusion": 0,
    "sepsis_or_meningitis": 0,
    "ear_anatomy_abnormality": 0,
    "oae_result": "pass",
    "aabr_result": "pass",
    "consent_for_research": 0
  }'
```

> Responses include the disclaimer string to reiterate that predictions are synthetic and non-diagnostic.

## Running the React Frontend

Install dependencies and start the development server:

```bash
cd frontend
npm install
npm start
```

The app expects the FastAPI backend on `http://localhost:8000`.

### Frontend Features

- Search bar to find patients by name (case-insensitive).
- Result card displaying patient metrics, probability, risk category, confidence, and a progress bar.
- New patient submission form with field validation and consent checkbox.
- Immediate display of prediction results after form submission.

## Data Privacy & Ethics

All data in this repository is synthetically generated. No personal information, audio, or video is used or produced. The dataset mimics clinical characteristics but is **entirely fictional**.

## Troubleshooting

- Ensure that you regenerate the dataset and retrain the model whenever the generation logic changes.
- If FastAPI cannot find the artifacts, confirm that `train_model.py` completed successfully and the `artifacts/` directory contains the required files.
- For Node.js certificate or proxy issues, consult your environment's documentation.

## License

This project is released for educational and research prototyping. Verify third-party package licenses before production use.
