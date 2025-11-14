# Data Schema Documentation

This document describes the CSV schemas used in the Hearing Deficiency ML project.

## Overview

The project uses three main CSV files:
1. **variants.csv** - Genomic variant data
2. **clinical.csv** - Clinical and demographic data
3. **features.csv** - Merged feature set for ML training

All data is synthetically generated with seed 42 for reproducibility.

## 1. Variants CSV (variants.csv)

Genomic variant annotations for hearing-loss related genes.

| Column | Type | Description | Example Values |
|--------|------|-------------|----------------|
| sample_id | string | Unique sample identifier | SAMPLE_0001 |
| gene | string | Gene symbol | GJB2, SLC26A4, OTOF, MYO7A, etc. |
| variant_id | string | Variant identifier | rs12345, chr13:20763612:G>A |
| chromosome | string | Chromosome | chr1, chr13, chrX |
| position | integer | Genomic position | 1234567 |
| ref | string | Reference allele | A, C, G, T |
| alt | string | Alternate allele | A, C, G, T |
| variant_type | string | Type of variant | SNV, INDEL, CNV |
| zygosity | string | Zygosity status | homozygous, heterozygous, compound_heterozygous |
| allele_frequency | float | Population allele frequency | 0.0001 to 0.05 |
| pathogenicity | string | Pathogenicity prediction | pathogenic, likely_pathogenic, benign, VUS |
| cadd_score | float | CADD deleteriousness score | 0 to 35 |
| impact | string | Functional impact | HIGH, MODERATE, LOW, MODIFIER |
| consequence | string | Variant consequence | missense_variant, frameshift, splice_site, etc. |

### Key Genes Included
- **GJB2** (connexin 26) - Most common genetic cause
- **SLC26A4** - Pendred syndrome
- **OTOF** - Auditory neuropathy
- **MYO7A, MYO15A** - Usher syndrome
- **STRC** - DFNB16
- **CDH23, PCDH15** - Usher syndrome type 1
- **TMC1, TMPRSS3** - Recessive hearing loss

## 2. Clinical CSV (clinical.csv)

Clinical, demographic, and perinatal data.

| Column | Type | Description | Example Values |
|--------|------|-------------|----------------|
| sample_id | string | Unique sample identifier (matches variants) | SAMPLE_0001 |
| age_months | integer | Age at screening | 0 to 36 months |
| sex | string | Biological sex | M, F |
| ethnicity | string | Ethnicity | Caucasian, Hispanic, Asian, African, Mixed |
| birth_weight_g | integer | Birth weight in grams | 500 to 5000 |
| gestational_age_weeks | float | Gestational age | 24 to 42 weeks |
| premature | integer | Born premature (< 37 weeks) | 0 (no), 1 (yes) |
| apgar_1min | integer | APGAR score at 1 minute | 0 to 10 |
| apgar_5min | integer | APGAR score at 5 minutes | 0 to 10 |
| nicu_days | integer | Days in NICU | 0 to 120 |
| mechanical_ventilation_days | integer | Days on ventilator | 0 to 60 |
| hyperbilirubinemia | integer | Elevated bilirubin | 0 (no), 1 (yes) |
| bilirubin_max_mg_dl | float | Maximum bilirubin level | 0 to 30 mg/dL |
| ototoxic_medications | integer | Exposure to ototoxic drugs | 0 (no), 1 (yes) |
| aminoglycoside_days | integer | Days of aminoglycoside exposure | 0 to 30 |
| loop_diuretic_days | integer | Days of loop diuretic exposure | 0 to 30 |
| maternal_cmv_infection | integer | Maternal CMV during pregnancy | 0 (no), 1 (yes) |
| maternal_rubella | integer | Maternal rubella during pregnancy | 0 (no), 1 (yes) |
| maternal_toxoplasmosis | integer | Maternal toxoplasmosis | 0 (no), 1 (yes) |
| family_history_hearing_loss | integer | Family history present | 0 (no), 1 (yes) |
| consanguinity | integer | Parental consanguinity | 0 (no), 1 (yes) |
| syndromic_features | integer | Presence of syndrome features | 0 (no), 1 (yes) |
| craniofacial_anomalies | integer | Craniofacial abnormalities | 0 (no), 1 (yes) |
| oae_result | string | OAE screening result | pass, refer, not_done |
| aabr_result | string | AABR screening result | pass, refer, not_done |
| diagnostic_abr_threshold_db | float | ABR threshold if performed | 20 to 100 dB HL |
| hearing_loss_severity | string | Severity classification | normal, mild, moderate, severe, profound |

## 3. Features CSV (features.csv)

Merged dataset combining genomic and clinical features for ML model training.

| Column | Type | Description | Source |
|--------|------|-------------|--------|
| sample_id | string | Unique identifier | Both |
| **Genomic Features** | | | |
| pathogenic_variant_count | integer | Count of pathogenic variants | variants.csv |
| has_gjb2_variant | integer | GJB2 variant present | variants.csv |
| has_slc26a4_variant | integer | SLC26A4 variant present | variants.csv |
| has_otof_variant | integer | OTOF variant present | variants.csv |
| max_cadd_score | float | Maximum CADD score | variants.csv |
| homozygous_variant_count | integer | Homozygous variant count | variants.csv |
| compound_het_count | integer | Compound heterozygous count | variants.csv |
| high_impact_variant_count | integer | High impact variants | variants.csv |
| **Clinical Features** | | | |
| age_months | integer | Age at screening | clinical.csv |
| sex | string | M/F encoded | clinical.csv |
| ethnicity | string | Ethnicity encoded | clinical.csv |
| birth_weight_g | integer | Birth weight | clinical.csv |
| gestational_age_weeks | float | Gestational age | clinical.csv |
| premature | integer | Premature birth flag | clinical.csv |
| apgar_1min | integer | APGAR 1 min | clinical.csv |
| apgar_5min | integer | APGAR 5 min | clinical.csv |
| nicu_days | integer | NICU days | clinical.csv |
| mechanical_ventilation_days | integer | Ventilation days | clinical.csv |
| hyperbilirubinemia | integer | Hyperbilirubinemia flag | clinical.csv |
| bilirubin_max_mg_dl | float | Max bilirubin | clinical.csv |
| ototoxic_medications | integer | Ototoxic meds flag | clinical.csv |
| aminoglycoside_days | integer | Aminoglycoside exposure | clinical.csv |
| maternal_cmv_infection | integer | CMV flag | clinical.csv |
| family_history_hearing_loss | integer | Family history flag | clinical.csv |
| consanguinity | integer | Consanguinity flag | clinical.csv |
| syndromic_features | integer | Syndrome flag | clinical.csv |
| craniofacial_anomalies | integer | Craniofacial flag | clinical.csv |
| oae_result | string | OAE result encoded | clinical.csv |
| aabr_result | string | AABR result encoded | clinical.csv |
| **Target Variable** | | | |
| hearing_impairment | integer | Binary target (0=normal, 1=impaired) | clinical.csv |
| severity_class | string | Multi-class target | clinical.csv |

## Data Generation Parameters

- **Sample Size**: 1000 samples minimum
- **Random Seed**: 42 (for reproducibility)
- **Class Balance**: Approximately 15-20% positive cases (hearing impairment)
- **Missing Data**: 2-5% random missingness to simulate real-world conditions
- **Realistic Distributions**: 
  - Birth weight: μ=3200g, σ=600g
  - Gestational age: μ=38.5 weeks, σ=2.5 weeks
  - APGAR scores: skewed toward higher values
  - Variant frequencies: realistic population frequencies

## Feature Engineering Notes

1. **Genomic Aggregations**: Variants are aggregated per sample to create gene-level and pathogenicity-level features
2. **Clinical Encodings**: Categorical variables (sex, ethnicity, screening results) are one-hot encoded
3. **Risk Factors**: High-risk clinical factors are weighted appropriately
4. **Interaction Features**: May be created during feature selection (e.g., genetic + environmental interactions)

## Data Quality Checks

The synthetic data generation ensures:
- No duplicate sample IDs
- Consistent data types
- Realistic value ranges
- Biological plausibility (e.g., premature flag matches gestational age)
- Label consistency with known risk factors

## References

This schema is inspired by:
1. Joint Committee on Infant Hearing 2019 Position Statement
2. ClinVar and gnomAD databases for variant pathogenicity
3. ACMG/AMP variant interpretation guidelines
4. Real-world EHDI program data structures
