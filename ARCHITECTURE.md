# System Architecture

## AI-Based Genomic Newborn Hearing Screening Platform

### Overview
This platform provides end-to-end genomic analysis for newborn hearing screening, integrating bioinformatics pipelines, AI/ML models, clinical reporting, and decision support systems.

---

## Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    External Interfaces                       │
│  (EHR Systems, LIMS, Clinical Portals via HL7 FHIR)        │
└─────────────────────────────────────────────────────────────┘
                            ↕
┌─────────────────────────────────────────────────────────────┐
│                      API Layer (FastAPI)                     │
│  • Sample Management  • Analysis Endpoints                   │
│  • Report Generation  • Health Checks                        │
└─────────────────────────────────────────────────────────────┘
                            ↕
┌─────────────────────────────────────────────────────────────┐
│                   Application Services                       │
├─────────────────┬───────────────┬──────────────────────────┤
│  Bioinformatics │   AI/ML       │   Reporting              │
│  Pipeline       │   Models      │   System                 │
│                 │               │                          │
│  • FASTQ QC     │  • Predictor  │  • PDF/JSON Generation   │
│  • Alignment    │  • Explainer  │  • EHR Integration       │
│  • Variant Call │  • Training   │  • Clinical Templates    │
│  • Annotation   │  • Evaluation │  • Audit Logging         │
│  • VCF Parsing  │  • Gene DB    │                          │
└─────────────────┴───────────────┴──────────────────────────┘
                            ↕
┌─────────────────────────────────────────────────────────────┐
│                    Data Layer (PostgreSQL)                   │
│  • Samples  • Analyses  • Variants  • Reports  • Audit Logs │
└─────────────────────────────────────────────────────────────┘
                            ↕
┌─────────────────────────────────────────────────────────────┐
│                   Infrastructure Layer                       │
│  • Docker/Kubernetes  • S3 Storage  • Encryption            │
│  • Authentication     • Logging     • Monitoring            │
└─────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. Bioinformatics Pipeline (`pipeline/`)

**Purpose**: Process raw sequencing data to produce annotated variants

**Components**:
- **FASTQ Processor**: Quality control and validation of raw reads
- **Alignment Module**: BWA-MEM interface for read mapping (hg38)
- **Variant Caller**: Identifies SNPs, indels from aligned reads
- **Annotator**: Integrates ClinVar, gnomAD, dbNSFP, OMIM annotations
- **VCF Parser**: Manipulates and queries VCF files

**Data Flow**:
```
FASTQ → QC → Alignment (BAM) → Variant Calling (VCF) → Annotation → Structured Output
```

**Key Technologies**: 
- BWA-MEM, SAMtools, BCFtools
- PyVCF, pysam
- ClinVar, gnomAD APIs

---

### 2. AI/ML Model (`model/`)

**Purpose**: Risk prediction and interpretation for hearing loss

**Components**:
- **Gene Database**: Curated list of 100+ hearing loss genes
  - GJB2 (Connexin 26), SLC26A4, OTOF, MYO7A, CDH23, etc.
- **Predictor**: Multi-level scoring (variant, gene, sample)
  - RandomForest, XGBoost, Neural Networks
- **Explainer**: SHAP-based interpretability
- **Training Pipeline**: Cross-validation, hyperparameter tuning
- **Evaluation**: ROC-AUC, Precision-Recall, Calibration

**Model Architecture**:
```
Input: Genomic Variants + Clinical Features
  ↓
Feature Engineering (PCA, ReliefF, Mutual Information)
  ↓
Ensemble Models (RF, XGBoost, ANN)
  ↓
Risk Score (0-1) + Confidence Interval
  ↓
SHAP Explanations (per variant contribution)
```

**Performance Targets**:
- AUC-ROC > 0.90
- Sensitivity > 0.95 (critical for screening)
- Specificity > 0.85

---

### 3. API Layer (`api/`)

**Purpose**: RESTful interface for all system operations

**Endpoints**:

**Samples**:
- `POST /api/samples` - Submit new sample
- `GET /api/samples/{id}` - Retrieve sample details
- `PUT /api/samples/{id}` - Update sample metadata

**Analysis**:
- `POST /api/analysis/predict` - Run risk prediction
- `GET /api/analysis/{id}` - Retrieve analysis results
- `POST /api/analysis/batch` - Batch processing

**Reports**:
- `GET /api/reports/{id}` - Generate clinical report (PDF/JSON)
- `POST /api/reports/ehr` - Send to EHR (HL7 FHIR)

**Health**:
- `GET /health` - System health check
- `GET /metrics` - Performance metrics

**Authentication**: JWT-based with role-based access control (RBAC)

**Middleware**:
- Request logging
- Error handling
- Rate limiting
- Input validation

---

### 4. Reporting System (`reporting/`)

**Purpose**: Generate clinical-grade reports for providers

**Features**:
- **PDF Reports**: CLIA/CAP compliant clinical reports
- **JSON Export**: Machine-readable structured data
- **EHR Integration**: HL7 FHIR R4 compliance
- **Templates**: Jinja2-based customizable templates

**Report Sections**:
1. Patient Demographics (de-identified)
2. Test Information
3. Risk Assessment Summary
4. Variant Details (pathogenic/likely pathogenic)
5. Gene-Level Analysis
6. Recommendations
7. Methodology & Limitations
8. References

---

### 5. Database Layer (`database/`)

**Purpose**: Persistent storage with ACID guarantees

**Schema**:

**Samples**:
- `id`, `external_id`, `birth_date`, `sex`, `ethnicity`
- `collection_date`, `sequencing_date`, `status`

**Analyses**:
- `id`, `sample_id`, `model_version`, `risk_score`
- `confidence`, `status`, `created_at`, `completed_at`

**Variants**:
- `id`, `analysis_id`, `chromosome`, `position`, `ref`, `alt`
- `gene`, `consequence`, `clinvar_sig`, `gnomad_af`
- `pathogenicity_score`

**Reports**:
- `id`, `analysis_id`, `format`, `content`, `generated_at`
- `recipient_id`, `sent_at`

**Audit Logs**:
- `id`, `user_id`, `action`, `resource`, `timestamp`, `ip_address`

**Technology**: PostgreSQL 15+ with pgcrypto for encryption at rest

---

### 6. Infrastructure (`infrastructure/`)

**Purpose**: Deployment, security, and observability

**Components**:
- **Docker**: Multi-stage builds for minimal images
- **Docker Compose**: Local development environment
- **Kubernetes**: Production orchestration (Helm charts)
- **S3 Storage**: Large file storage (FASTQ, BAM, VCF)
- **Encryption**: AES-256 for data at rest, TLS 1.3 for transit
- **Authentication**: OAuth2/OIDC integration
- **Logging**: Structured JSON logs (ELK stack)
- **Monitoring**: Prometheus + Grafana

**Security Considerations**:
- HIPAA compliance: PHI encryption, audit logs, access controls
- GDPR compliance: Data minimization, right to erasure
- Penetration testing: Regular security audits
- Secrets management: HashiCorp Vault or AWS Secrets Manager

---

### 7. Utilities (`utils/`)

**Purpose**: Cross-cutting concerns and helper functions

**Modules**:
- **Logging**: Structured logging with correlation IDs
- **Validators**: Input validation (VCF format, sample IDs)
- **Constants**: Application-wide constants (genes, thresholds)
- **Helpers**: Common utilities (date parsing, file handling)

---

## Data Flow: End-to-End

```
1. Sample Collection
   ↓
2. Sequencing (External Lab) → FASTQ files
   ↓
3. Upload to Platform (API) → S3 Storage
   ↓
4. Bioinformatics Pipeline
   - Quality Control
   - Alignment (hg38)
   - Variant Calling
   - Annotation (ClinVar, gnomAD)
   ↓
5. Structured Variant Data → Database
   ↓
6. AI/ML Prediction
   - Feature Engineering
   - Ensemble Models
   - Risk Score Calculation
   - SHAP Explanations
   ↓
7. Analysis Results → Database
   ↓
8. Report Generation
   - Clinical PDF Report
   - JSON Export
   - EHR Integration (HL7 FHIR)
   ↓
9. Clinician Review → Patient Management
```

---

## Scalability & Performance

**Throughput Targets**:
- 1,000 samples/day
- < 10 minute analysis time per sample
- 99.9% API uptime

**Scaling Strategies**:
- Horizontal scaling: Kubernetes auto-scaling
- Async processing: Celery task queue
- Caching: Redis for frequently accessed data
- Database: Read replicas for queries

---

## Testing Strategy

**Unit Tests**: 80%+ coverage per module
**Integration Tests**: End-to-end pipeline validation
**API Tests**: Contract testing with OpenAPI schemas
**Performance Tests**: Load testing with Locust
**Security Tests**: OWASP Top 10 validation

---

## Deployment Environments

**Development**: Docker Compose on local machines
**Staging**: Kubernetes cluster with synthetic data
**Production**: Kubernetes cluster with real data
- Multi-region deployment
- Blue-green deployment strategy
- Automated rollback on failures

---

## Compliance & Regulations

**Clinical Validation**: CAP/CLIA laboratory requirements
**Data Privacy**: HIPAA, GDPR, state-specific laws
**Quality Management**: ISO 13485 medical devices
**Bioinformatics Standards**: GA4GH, HL7 FHIR

---

## Future Enhancements

1. **Pharmacogenomics**: Drug-gene interactions for ototoxicity
2. **Multi-omics**: Integrate transcriptomics, proteomics
3. **Federated Learning**: Train on distributed datasets
4. **Real-time Analysis**: Stream processing for rapid turnaround
5. **Clinical Decision Support**: Integration with EHR workflows
6. **Population Studies**: Large-scale epidemiological analysis

---

## Technology Stack Summary

| Layer | Technology |
|-------|------------|
| API | FastAPI, Pydantic |
| ML/AI | Scikit-learn, XGBoost, PyTorch, SHAP |
| Database | PostgreSQL, SQLAlchemy |
| Bioinformatics | PyVCF, pysam, BWA-MEM |
| Reporting | Jinja2, WeasyPrint, HL7 FHIR |
| Infrastructure | Docker, Kubernetes, AWS/GCP |
| Testing | Pytest, Locust, OpenAPI |
| Monitoring | Prometheus, Grafana, ELK |

---

**Document Version**: 1.0  
**Last Updated**: 2025-12-03  
**Maintained By**: Platform Architecture Team
