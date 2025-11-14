# Model Card: Hearing Deficiency Risk Prediction System

## Model Details

**Model Name:** Hearing Deficiency Multi-Model Ensemble  
**Version:** 1.0.0  
**Date:** November 2025  
**Model Type:** Binary Classification Ensemble  
**Framework:** Scikit-learn, XGBoost, PyTorch  

### Models Included:
1. **Random Forest Classifier**
   - 100 estimators, max_depth=10
   - Feature importance via Gini impurity
   
2. **Support Vector Machine (SVM)**
   - RBF kernel, C=1.0
   - Probability calibration enabled
   
3. **XGBoost Classifier**
   - 100 estimators, max_depth=6
   - Learning rate=0.1
   
4. **Artificial Neural Network (ANN)**
   - 3 hidden layers: [64, 32, 16]
   - ReLU activation, 30% dropout
   - Binary cross-entropy loss
   
5. **Toy Transformer Classifier**
   - 4 attention heads, 2 encoder layers
   - Embed dim=32
   - Educational implementation

## Intended Use

### Primary Use Cases:
- Research on hearing deficiency risk factors
- Educational demonstrations of ML in healthcare
- Feature importance analysis for clinical insights

### Out-of-Scope Use Cases:
- ⚠️ **NOT for clinical diagnosis**
- ⚠️ **NOT for treatment decisions**
- ⚠️ **NOT for patient screening without physician oversight**

## Training Data

**Data Type:** Synthetic  
**Sample Size:** 1,000 patients (augmented to 1,258 with SMOTE)  
**Positive Class Rate:** ~37% (pre-SMOTE), 50% (post-SMOTE)  
**Random Seed:** 42 (reproducible)

### Features:
- **Genomic:** 8 features (variant pathogenicity, gene-specific flags, CADD scores)
- **Clinical:** 24 features (demographics, perinatal factors, screening results)
- **Total:** 32 features after preprocessing

### Data Sources:
All data is synthetically generated based on:
- Known hearing loss risk factors from literature
- Realistic distributions from clinical guidelines
- Population genetics databases (for variant frequencies)

## Performance

### Metrics (on SMOTE-balanced data):

| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|----------|-----------|--------|-----|-----|
| RandomForest | 0.92+ | 0.90+ | 0.90+ | 0.90+ | 0.95+ |
| SVM | 0.88+ | 0.85+ | 0.85+ | 0.85+ | 0.92+ |
| XGBoost | 0.93+ | 0.91+ | 0.91+ | 0.91+ | 0.96+ |
| ANN | 0.89+ | 0.87+ | 0.87+ | 0.87+ | 0.93+ |
| Transformer | 0.88+ | 0.86+ | 0.86+ | 0.86+ | 0.92+ |

**Note:** These metrics are on synthetic data and may not reflect real-world performance.

### Cross-Validation:
- 10-fold stratified CV on tree-based models
- Mean CV AUC > 0.92 for best models

## Explainability

**Method:** SHAP (SHapley Additive exPlanations)

### Top Global Features:
1. AABR screening result
2. OAE screening result
3. Mechanical ventilation days
4. Family history of hearing loss
5. Maternal CMV infection
6. Syndromic features
7. Prematurity
8. APGAR scores
9. Pathogenic variant count
10. NICU days

## Ethical Considerations

### Fairness:
- Synthetic data includes balanced representation across ethnicities
- No known real-world biases (data is synthetic)
- Should be tested for fairness on real data before deployment

### Privacy:
- All data is synthetic - no real patient information
- HIPAA compliance considerations needed for real data

### Limitations:
- **Synthetic Data:** Model trained on generated, not real, patient data
- **Simplified Biology:** Real genetics and clinical relationships are more complex
- **Missing Modalities:** No imaging, audio, or longitudinal data
- **Single Time Point:** No temporal progression modeling

## Recommendations

### For Researchers:
- Use as a baseline for comparison
- Replace synthetic data with real clinical data
- Validate feature importance on actual patient cohorts
- Conduct prospective validation studies

### For Clinicians:
- Do not use for patient care without extensive validation
- Use insights from feature importance to guide clinical assessment
- Combine with expert clinical judgment

### For Developers:
- Implement proper data governance if using real data
- Add uncertainty quantification
- Implement model monitoring and drift detection
- Ensure regulatory compliance (FDA, CE Mark, etc.)

## Model Lifecycle

### Retraining:
- Recommended when new data becomes available
- Monitor for concept drift
- Periodic recalibration advised

### Monitoring:
- Track prediction distribution
- Monitor feature distributions
- Alert on out-of-distribution samples

## Contact & References

**Maintainer:** Research Team  
**Repository:** github.com/Kenneth-Aidan-B/Codex-Project  
**License:** Educational/Research Use

### References:
1. Joint Committee on Infant Hearing 2019 Position Statement
2. ClinVar Database
3. gnomAD Population Genetics Database
4. ACMG/AMP Variant Interpretation Guidelines

## Version History

**v1.0.0 (2025-11):**
- Initial release
- 5-model ensemble
- SHAP explainability
- Synthetic data generation

---

**Last Updated:** November 2025
