# Data Dictionary

## Overview
This document describes all features and variables used in the hearing deficiency prediction models.

## Target Variable

| Variable | Type | Description | Values |
|----------|------|-------------|--------|
| `hearing_impairment` | Binary | Presence of hearing loss | 0 = Normal, 1 = Impaired |
| `hearing_loss_severity` | Categorical | Severity classification | normal, mild, moderate, severe, profound |

## Clinical Features

### Demographics
| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `age_months` | Integer | 0-36 | Age at screening in months |
| `sex` | Categorical | M, F | Biological sex |
| `ethnicity` | Categorical | 5 categories | Caucasian, Hispanic, Asian, African, Mixed |

### Birth Characteristics
| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `birth_weight_g` | Integer | 500-5000 | Birth weight in grams |
| `gestational_age_weeks` | Float | 24-42 | Gestational age at birth |
| `premature` | Binary | 0, 1 | Born before 37 weeks |
| `apgar_1min` | Integer | 0-10 | APGAR score at 1 minute |
| `apgar_5min` | Integer | 0-10 | APGAR score at 5 minutes |

### NICU and Complications
| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `nicu_days` | Integer | 0-120 | Days in neonatal intensive care |
| `mechanical_ventilation_days` | Integer | 0-60 | Days on mechanical ventilation |
| `hyperbilirubinemia` | Binary | 0, 1 | Elevated bilirubin levels |
| `bilirubin_max_mg_dl` | Float | 0-30 | Maximum bilirubin level (mg/dL) |

### Medication Exposure
| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `ototoxic_medications` | Binary | 0, 1 | Exposure to ototoxic drugs |
| `aminoglycoside_days` | Integer | 0-30 | Days of aminoglycoside antibiotics |
| `loop_diuretic_days` | Integer | 0-30 | Days of loop diuretics |

### Maternal Risk Factors
| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `maternal_cmv_infection` | Binary | 0, 1 | Maternal cytomegalovirus infection |
| `maternal_rubella` | Binary | 0, 1 | Maternal rubella during pregnancy |
| `maternal_toxoplasmosis` | Binary | 0, 1 | Maternal toxoplasmosis |

### Family History and Syndromes
| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `family_history_hearing_loss` | Binary | 0, 1 | Family history of hearing loss |
| `consanguinity` | Binary | 0, 1 | Parental consanguinity |
| `syndromic_features` | Binary | 0, 1 | Presence of syndrome features |
| `craniofacial_anomalies` | Binary | 0, 1 | Craniofacial abnormalities |

### Screening Results
| Feature | Type | Values | Description |
|---------|------|--------|-------------|
| `oae_result` | Categorical | pass, refer, not_done | Otoacoustic emissions test |
| `aabr_result` | Categorical | pass, refer, not_done | Automated ABR test |
| `diagnostic_abr_threshold_db` | Float | 20-100 | Diagnostic ABR threshold (dB HL) |

## Genomic Features

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `pathogenic_variant_count` | Integer | 0-5 | Number of pathogenic/likely pathogenic variants |
| `has_gjb2_variant` | Binary | 0, 1 | Variant in GJB2 gene (connexin 26) |
| `has_slc26a4_variant` | Binary | 0, 1 | Variant in SLC26A4 gene (Pendred syndrome) |
| `has_otof_variant` | Binary | 0, 1 | Variant in OTOF gene (auditory neuropathy) |
| `max_cadd_score` | Float | 0-35 | Maximum CADD deleteriousness score |
| `homozygous_variant_count` | Integer | 0-3 | Number of homozygous variants |
| `compound_het_count` | Integer | 0-2 | Number of compound heterozygous variants |
| `high_impact_variant_count` | Integer | 0-3 | Number of high-impact variants |

## Derived Features (Post-Preprocessing)

After preprocessing, features undergo:
- **Encoding:** Categorical variables (sex, ethnicity, screening results) are label-encoded
- **Scaling:** Numerical features are standardized (mean=0, std=1)
- **Imputation:** Missing values filled with median (numerical) or mode (categorical)

## Feature Importance

Based on SHAP analysis, features are ranked by their impact on predictions:

### Top 10 Most Important Features:
1. **aabr_result** - Automated ABR screening result
2. **oae_result** - OAE screening result
3. **mechanical_ventilation_days** - Duration of mechanical ventilation
4. **family_history_hearing_loss** - Family history
5. **maternal_cmv_infection** - Maternal CMV
6. **syndromic_features** - Presence of syndrome
7. **premature** - Premature birth
8. **apgar_5min** - 5-minute APGAR score
9. **pathogenic_variant_count** - Number of pathogenic variants
10. **nicu_days** - NICU duration

## Missing Data Handling

- **Rate:** 0-2% random missingness in synthetic data
- **Strategy:** 
  - Numerical: Median imputation
  - Categorical: Mode imputation or 'unknown' category

## Class Imbalance

- **Original Distribution:** ~63% negative, ~37% positive
- **Balancing Method:** SMOTE (Synthetic Minority Over-sampling Technique)
- **Final Distribution:** 50% negative, 50% positive

## Data Quality Checks

All data undergoes validation:
- No duplicate sample IDs
- Values within expected ranges
- Biologically plausible relationships
- Consistent encoding schemes

## References

- Joint Committee on Infant Hearing 2019 Position Statement
- ClinVar pathogenicity classifications
- gnomAD allele frequency data
- ACMG/AMP variant interpretation guidelines
