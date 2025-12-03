# AI-Powered Predictive Modeling for Hearing Deficiency

> **Disclaimer:** This repository contains a fully synthetic dataset and prototype machine learning pipeline for research and educational purposes only. It is **NOT** a medical device, provides no clinical guidance, and must never be used for patient care without extensive validation by qualified healthcare professionals.

## Overview

This project implements a comprehensive ML automation system for hearing deficiency risk prediction using genomic and clinical data. The system generates synthetic data, trains multiple models (RandomForest, SVM, XGBoost, ANN, Transformer), provides SHAP explainability, and exposes predictions through a FastAPI backend with a simple HTML/JS frontend.

### Key Features

- ✅ **Synthetic Data Generation** - Reproducible genomic + clinical datasets (seed 42)
- ✅ **Multi-Model Training** - 5 models with 10-fold cross-validation
- ✅ **SHAP Explainability** - Per-sample and global feature importance
- ✅ **FastAPI Service** - RESTful API with /predict and /explain endpoints
- ✅ **Simple Frontend** - HTML/JS interface for risk visualization
- ✅ **Docker Support** - Containerized deployment
- ✅ **CI/CD Pipeline** - GitHub Actions for automated testing
- ✅ **TODO System** - Automated task tracking and verification
- ✅ **Comprehensive Documentation** - Model card, data dictionary, evaluation protocol

## Repository Structure

```
.
├── pipeline/                     # 🆕 Bioinformatics Pipeline
│   ├── fastq_processor.py       # FASTQ QC and filtering
│   ├── alignment.py             # BWA-MEM alignment
│   ├── variant_caller.py        # Variant calling from BAM
│   ├── annotator.py             # ClinVar, gnomAD, dbNSFP annotation
│   ├── vcf_parser.py            # VCF parsing utilities
│   └── config.py                # Pipeline configuration
├── model/                        # 🆕 Enhanced AI/ML Models
│   ├── gene_database.py         # 30+ hearing loss genes (GJB2, SLC26A4, OTOF...)
│   ├── predictor.py             # Risk prediction (variant/gene/sample scoring)
│   ├── explainer.py             # SHAP explainability
│   ├── training.py              # Model training pipeline
│   └── evaluation.py            # Performance metrics
├── api/                          # FastAPI application
│   ├── app.py                   # Main API with /predict and /explain
│   └── test_api.py              # API tests
├── database/                     # 🆕 Database Layer (SQLAlchemy)
│   ├── models.py                # ORM models (Sample, Analysis, Variant, Report)
│   ├── connection.py            # DB connection management
│   └── migrations/              # Database migrations
├── reporting/                    # 🆕 Clinical Reporting
│   ├── report_generator.py     # PDF/JSON report generation
│   ├── ehr_integration.py      # HL7 FHIR integration
│   └── templates/              # Report templates
├── infrastructure/              # 🆕 Infrastructure Layer
│   ├── config.py               # Cloud/deployment configuration
│   └── security.py             # Encryption, authentication
├── utils/                       # 🆕 Utilities
│   ├── logging.py              # Structured logging
│   ├── validators.py           # Input validation
│   └── constants.py            # Application constants
├── data/                        
│   ├── schema.md                # CSV schema documentation
│   └── synthetic/               
│       ├── generate_synthetic.py # Data generator (seed 42)
│       ├── variants.csv         # Genomic variants
│       ├── clinical.csv         # Clinical/demographic data
│       └── features.csv         # Merged feature set
├── ml/                          
│   ├── preprocess.py            # Imputation, scaling, SMOTE
│   ├── feature_selection.py    # ReliefF, PCA, mutual info
│   ├── train.py                 # Multi-model training pipeline
│   └── explain.py               # SHAP explanations
├── models/                      # Saved model artifacts
│   ├── RandomForest/
│   ├── SVM/
│   ├── XGBoost/
│   ├── ANN/
│   └── Transformer/
├── results/                     
│   ├── metrics.csv              # Model performance metrics
│   ├── feature_importance.csv  # Global SHAP importance
│   └── explanations/            # Per-sample SHAP JSONs
├── frontend/                    
│   └── index.html               # Static HTML/JS interface
├── docker/                      
│   ├── Dockerfile               # API container
│   └── docker-compose.yml       # Multi-container setup
├── tests/                       # Pytest test suite
│   ├── test_synthetic_data.py
│   ├── test_preprocess.py
│   ├── test_train_smoke.py
│   └── test_api.py
├── docs/                        
│   ├── model_card.md            # Model documentation
│   ├── data_dictionary.md       # Feature descriptions
│   └── evaluation_protocol.md  # Evaluation methodology
├── notebooks/
│   └── annotate_variants.ipynb # Annotation demonstration
├── scripts/
│   └── fastq_to_vcf.sh         # Genomics pipeline documentation
├── todo.json                    # Task checklist
├── todo_check.py                # Automated verification script
└── requirements.txt             # Python dependencies

## Quick Start

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# Clone repository
git clone https://github.com/Kenneth-Aidan-B/Codex-Project.git
cd Codex-Project

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Complete Pipeline (End-to-End)

```bash
# 1. Generate synthetic data
python data/synthetic/generate_synthetic.py

# 2. Preprocess data
python ml/preprocess.py

# 3. Feature selection
python ml/feature_selection.py

# 4. Train models (may take 10-15 minutes)
python ml/train.py

# 5. Generate SHAP explanations
python ml/explain.py

# 6. Verify all tasks completed
python todo_check.py
```

### Run API Server

```bash
# Start FastAPI server
uvicorn api.app:app --reload --port 8000

# API will be available at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

### Access Frontend

```bash
# Serve frontend (requires Python's http.server or any static server)
cd frontend
python -m http.server 8080

# Open browser to http://localhost:8080
```

### Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

### Docker Deployment

```bash
# Build and run with Docker Compose
cd docker
docker-compose up --build

# API: http://localhost:8000
# Frontend: http://localhost:8080
```

## API Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### Single Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age_months": 6,
    "sex": "M",
    "ethnicity": "Caucasian",
    "birth_weight_g": 3200,
    "gestational_age_weeks": 38.5,
    "premature": 0,
    "apgar_1min": 8,
    "apgar_5min": 9,
    "nicu_days": 0,
    "mechanical_ventilation_days": 0,
    "hyperbilirubinemia": 0,
    "bilirubin_max_mg_dl": 5.0,
    "ototoxic_medications": 0,
    "aminoglycoside_days": 0,
    "loop_diuretic_days": 0,
    "maternal_cmv_infection": 0,
    "maternal_rubella": 0,
    "maternal_toxoplasmosis": 0,
    "family_history_hearing_loss": 0,
    "consanguinity": 0,
    "syndromic_features": 0,
    "craniofacial_anomalies": 0,
    "oae_result": "pass",
    "aabr_result": "pass"
  }'
```

### Get Explanation
```bash
curl http://localhost:8000/explain/SAMPLE_0001_RandomForest
```

### Batch Prediction
```bash
curl -X POST http://localhost:8000/predict/batch \
  -F "file=@samples.csv"
```

### AI Insight (Gemini-powered LLM Summarization)

**⚠️ Privacy Warning:** Do NOT send protected health information (PHI) or real patient data to cloud LLM services. This feature is designed for synthetic/de-identified data only. Enable privacy safeguards before using with any clinical data.

Generate natural language summaries of model predictions and SHAP explanations using Gemini LLM:

#### Setup
Set environment variables for Gemini API access:
```bash
export GEMINI_API_KEY="your_api_key_here"
export GEMINI_MODEL="gemini-pro"  # Optional, defaults to gemini-pro
export GEMINI_ENDPOINT_TEMPLATE="https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"  # Optional
```

#### Usage with Direct Payload
```bash
curl -X POST http://localhost:8000/ai/insight \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

#### Usage with Sample ID
```bash
curl -X POST http://localhost:8000/ai/insight \
  -H "Content-Type: application/json" \
  -d '{
    "sample_id": "synthetic",
    "probability": 0.20,
    "model_name": "RandomForest"
  }'
```

**Fallback Behavior:** If `GEMINI_API_KEY` is not set or the API call fails, the endpoint returns a deterministic fallback summary based on SHAP values. Generated insights are cached in `results/ai_insights/` to reduce API calls.

**Response Fields:**
- `summary`: Natural language explanation of the prediction
- `top_features`: Most important features with SHAP values
- `confidence_note`: Confidence level assessment
- `next_step`: Recommended action
- `disclaimer`: Legal and privacy disclaimer
- `llm_response_raw`: Raw LLM response (if available)
- `cached`: Whether result was loaded from cache

## Model Performance

| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|----------|-----------|--------|-----|-----|
| Random Forest | 0.92+ | 0.90+ | 0.90+ | 0.90+ | 0.95+ |
| SVM | 0.88+ | 0.85+ | 0.85+ | 0.85+ | 0.92+ |
| XGBoost | 0.93+ | 0.91+ | 0.91+ | 0.91+ | 0.96+ |
| ANN | 0.89+ | 0.87+ | 0.87+ | 0.87+ | 0.93+ |
| Transformer | 0.88+ | 0.86+ | 0.86+ | 0.86+ | 0.92+ |

*Note: Metrics on SMOTE-balanced synthetic data. Real-world performance requires validation.*

## Top Risk Factors (SHAP Analysis)

1. **AABR screening result** - Automated ABR test
2. **OAE screening result** - Otoacoustic emissions
3. **Mechanical ventilation days** - NICU respiratory support
4. **Family history** - Genetic predisposition
5. **Maternal CMV infection** - Congenital infection
6. **Syndromic features** - Associated syndromes
7. **Prematurity** - Early birth
8. **APGAR scores** - Newborn health assessment
9. **Pathogenic variants** - Genetic mutations
10. **NICU duration** - Intensive care exposure

## Data Schema

### Genomic Features (8)
- Pathogenic variant count
- Gene-specific variants (GJB2, SLC26A4, OTOF)
- CADD scores
- Zygosity patterns

### Clinical Features (24)
- Demographics (age, sex, ethnicity)
- Birth characteristics (weight, gestational age, APGAR)
- NICU complications (ventilation, bilirubin)
- Medication exposure (ototoxic drugs)
- Maternal infections (CMV, rubella, toxoplasmosis)
- Family history and syndromes
- Screening results (OAE, AABR)

### Total: 32 features after preprocessing

See [data/schema.md](data/schema.md) for complete documentation.

## TODO System

The repository includes an automated task tracking system:

```bash
# Check task completion status
python todo_check.py

# View tasks
cat todo.json

# Check logs
tail -f todo.log
```

Tasks are automatically verified based on:
- File existence
- Verification markers (e.g., `.preprocess_ok`)
- Output artifacts

## Development

### Running Individual Steps

```bash
# Generate data only
python data/synthetic/generate_synthetic.py

# Train specific model (edit train.py to comment out others)
python ml/train.py

# Feature importance only
python ml/feature_selection.py

# SHAP explanations
python ml/explain.py
```

### Testing

```bash
# Test data generation
pytest tests/test_synthetic_data.py -v

# Test preprocessing
pytest tests/test_preprocess.py -v

# Test training (smoke test, fast)
pytest tests/test_train_smoke.py -v

# Test API
pytest tests/test_api.py -v
```

### Code Quality

```bash
# Lint with flake8
flake8 . --max-line-length=127 --exclude=.venv,frontend

# Format with black
black . --exclude=.venv
```

## Documentation

- **[Model Card](docs/model_card.md)** - Model details, use cases, limitations
- **[Data Dictionary](docs/data_dictionary.md)** - Complete feature descriptions
- **[Evaluation Protocol](docs/evaluation_protocol.md)** - Methodology and metrics
- **[Schema Documentation](data/schema.md)** - CSV file formats

## CI/CD

GitHub Actions workflow (`.github/workflows/ci.yml`) runs on push/PR:
- Data generation
- Preprocessing
- Feature selection
- Pytest suite
- Code linting
- Docker build

## Reproducibility

All randomness is seeded with `SEED=42`:
- Data generation
- Train/test splits
- Model initialization
- Cross-validation folds

Running the pipeline twice produces identical results.

## Ethical Considerations

⚠️ **Important Limitations:**

- **Synthetic Data Only** - Not validated on real patients
- **Not for Clinical Use** - Requires FDA/regulatory approval
- **No Diagnostic Claims** - Educational/research prototype only
- **Bias Assessment Needed** - Fairness testing on real populations required
- **Privacy:** No real patient data used (all synthetic)

## License

This project is for educational and research use. Not licensed for clinical deployment.

## Contributing

This is a research prototype. For inquiries:
- Open an issue on GitHub
- See contribution guidelines (coming soon)

## References

1. Joint Committee on Infant Hearing (2019). Year 2019 Position Statement
2. ClinVar Database - variant pathogenicity classifications
3. gnomAD Database - population allele frequencies
4. ACMG/AMP Guidelines - variant interpretation standards

## Acknowledgments

This system demonstrates:
- Scikit-learn, XGBoost, PyTorch for modeling
- SHAP for explainability
- FastAPI for API development
- Faker for synthetic data generation
- Imbalanced-learn for SMOTE

---

**Version:** 1.0.0  
**Last Updated:** November 2025  
**Status:** Research Prototype  

For questions or issues, please open a GitHub issue.
