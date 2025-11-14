# Evaluation Protocol

## Overview
This document describes the evaluation methodology for the hearing deficiency prediction models.

## Evaluation Strategy

### Data Splitting
- **Training Set:** 80% of balanced data (after SMOTE)
- **Test Set:** 20% held-out data
- **Validation:** 10-fold stratified cross-validation on training set

### Balancing
- **Method:** SMOTE (Synthetic Minority Over-sampling Technique)
- **Reason:** Address class imbalance (~37% positive class)
- **Result:** 50-50 class distribution for training

## Performance Metrics

### Primary Metrics

1. **Area Under ROC Curve (AUC)**
   - **Why:** Class-balanced metric, threshold-independent
   - **Target:** > 0.90
   - **Interpretation:** Model's ability to distinguish classes

2. **F1 Score**
   - **Why:** Harmonic mean of precision and recall
   - **Target:** > 0.85
   - **Interpretation:** Balance between false positives and false negatives

### Secondary Metrics

3. **Accuracy**
   - Overall correctness
   - Less important due to class imbalance

4. **Precision**
   - Positive predictive value
   - Important for avoiding false alarms

5. **Recall (Sensitivity)**
   - True positive rate
   - Critical for not missing actual cases

6. **Specificity**
   - True negative rate
   - Important for reducing unnecessary interventions

## Cross-Validation

### Method: Stratified K-Fold
- **K:** 10 folds
- **Stratification:** Maintains class distribution in each fold
- **Randomization:** Seed=42 for reproducibility

### Reporting
- Mean ± Standard deviation across folds
- Individual fold performance
- Consistency across folds (low std indicates robustness)

## Model Comparison

### Ensemble Approach
Five models trained independently:
1. Random Forest
2. Support Vector Machine
3. XGBoost/Gradient Boosting
4. Artificial Neural Network
5. Transformer (toy implementation)

### Comparison Criteria
- AUC-ROC curves
- Precision-Recall curves
- Confusion matrices
- Training time
- Inference time
- Model interpretability

### Best Model Selection
- Highest cross-validated AUC
- Consistency across folds (low variance)
- Interpretability (SHAP support)
- **Default:** Random Forest (good balance)

## Explainability Evaluation

### SHAP Analysis
- **Global Importance:** Average |SHAP value| across all samples
- **Local Explanations:** Per-sample SHAP values
- **Consistency:** Agreement with domain knowledge

### Feature Importance Validation
- Clinical relevance of top features
- Comparison with literature
- Biological plausibility

## Calibration Assessment

### Reliability Diagrams
- Plot predicted probabilities vs. actual frequencies
- Assess calibration error

### Brier Score
- Mean squared difference between predicted probabilities and outcomes
- Lower is better

## Robustness Testing

### Perturbation Analysis
- Small changes to input features
- Check prediction stability

### Out-of-Distribution Detection
- Test with edge cases
- Identify when model should abstain

## Fairness Evaluation

### Subgroup Analysis
Evaluate performance across:
- **Ethnicity:** Ensure no bias
- **Sex:** Check for gender disparities
- **Gestational Age:** Preterm vs. term
- **Birth Weight:** Low vs. normal

### Metrics by Subgroup
- AUC, precision, recall for each group
- Statistical testing for significant differences
- Equalized odds assessment

## Error Analysis

### Misclassification Review
- Identify common patterns in errors
- False positives: Low-risk predicted as high-risk
- False negatives: High-risk missed

### Feature Analysis of Errors
- Which features contribute to errors?
- Are there systematic patterns?

## Real-World Considerations

### Prospective Validation (Future)
- Deploy in clinical setting
- Monitor predictions vs. actual outcomes
- Continuous recalibration

### Temporal Validation
- Train on older data
- Test on newer data
- Assess temporal stability

## Reporting Standards

### Minimum Requirements
1. Sample sizes (train/test/CV)
2. All performance metrics with confidence intervals
3. ROC and PR curves
4. Confusion matrices
5. Feature importance rankings
6. Subgroup performance
7. Error analysis summary

### Reproducibility
- Random seed documented (42)
- Exact data splits saved
- Preprocessing steps recorded
- Model hyperparameters logged

## Benchmarks

### Comparison Baselines
1. **Random Classifier:** AUC = 0.5
2. **Majority Class:** Accuracy = 0.63 (pre-balance)
3. **Clinical Guidelines:** Existing risk stratification tools

### Success Criteria
- AUC > 0.85 (good)
- AUC > 0.90 (excellent)
- F1 > 0.80
- Balanced performance across subgroups

## Limitations

### Current Evaluation Limitations
- **Synthetic Data:** Not validated on real patients
- **Single Time Point:** No longitudinal validation
- **Limited Diversity:** Synthetic data may not capture all real-world variation
- **No External Validation:** Only internal cross-validation

### Future Validation Needs
- External dataset testing
- Multi-center studies
- Prospective clinical trials
- Long-term outcome tracking

## Continuous Monitoring (Production)

### Model Drift Detection
- Monitor feature distributions
- Track prediction distributions
- Alert on significant changes

### Performance Tracking
- Real-time metric computation
- Periodic re-evaluation
- Feedback loop for improvements

## Ethical Considerations

### Evaluation Ethics
- Ensure fairness across demographics
- Transparent reporting of limitations
- Clear communication of uncertainty
- No deployment without clinical validation

---

**Last Updated:** November 2025  
**Version:** 1.0.0
