# Data Schema Documentation

This document describes the CSV schemas used in the **Genetic** Hearing Deficiency ML project.

## Overview

The project uses a **genetics-only** approach with three main CSV files:
1. **variants.csv** - Genomic variant data
2. **genetic_profiles.csv** - Genetic profile and family history data
3. **features.csv** - Merged genetic feature set for ML training

All data is synthetically generated with seed 42 for reproducibility.

---

## 1. Variants CSV (variants.csv)

Genomic variant annotations for hearing-loss related genes.

| Column | Type | Description | Example Values |
|--------|------|-------------|----------------|
| sample_id | string | Unique sample identifier | SAMPLE_0001 |
| gene | string | Gene symbol | GJB2, SLC26A4, OTOF, MYO7A, etc. |
| variant_id | string | Variant identifier | chr13:20763612:G>A |
| chromosome | string | Chromosome | chr1, chr13, chrX |
| position | integer | Genomic position | 1234567 |
| ref | string | Reference allele | A, C, G, T |
| alt | string | Alternate allele | A, C, G, T |
| variant_type | string | Type of variant | SNV, INDEL, CNV |
| zygosity | string | Zygosity status | homozygous, heterozygous, compound_heterozygous |
| allele_frequency | float | Population allele frequency | 0.0001 to 0.05 |
| pathogenicity | string | Pathogenicity prediction | pathogenic, likely_pathogenic, benign, VUS |
| cadd_score | float | CADD deleteriousness score | 0 to 40 |
| revel_score | float | REVEL pathogenicity score | 0 to 1 |
| impact | string | Functional impact | HIGH, MODERATE, LOW, MODIFIER |
| consequence | string | Variant consequence | missense_variant, frameshift_variant, splice_site, etc. |
| inheritance_pattern | string | Gene inheritance pattern | recessive, dominant, both |

### Key Genes Analyzed
- **GJB2** (Connexin 26) - Most common genetic cause (~50%)
- **SLC26A4** (Pendrin) - Pendred syndrome (~10%)
- **OTOF** (Otoferlin) - Auditory neuropathy (2-3%)
- **MYO7A** (Myosin VIIA) - Usher syndrome (3-5%)
- **CDH23** (Cadherin 23) - Usher syndrome type 1D (2-3%)
- **TMC1** - DFNB7/11 (1-2%)
- **MYO15A, STRC, PCDH15, TMPRSS3, KCNQ4, TECTA, WFS1, TRIOBP**

---

## 2. Genetic Profiles CSV (genetic_profiles.csv)

Genetic background and family history data.

| Column | Type | Description | Example Values |
|--------|------|-------------|----------------|
| sample_id | string | Unique sample identifier | SAMPLE_0001 |
| ethnicity | string | Ethnicity | Caucasian, Hispanic, Asian, African, Middle_Eastern, Mixed |
| family_history_hearing_loss | integer | Family history present | 0 (no), 1 (yes) |
| consanguinity | integer | Parental consanguinity | 0 (no), 1 (yes) |
| syndromic_genetic_condition | integer | Syndromic condition present | 0 (no), 1 (yes) |
| mtdna_variant_detected | integer | Mitochondrial DNA variant | 0 (no), 1 (yes) |
| polygenic_risk_score | float | Polygenic risk score | 0.0 to 1.0 |
| gjb2_carrier | integer | GJB2 carrier status | 0 (no), 1 (yes) |
| num_affected_relatives | integer | Number of affected relatives | 0 to 10 |

---

## 3. Features CSV (features.csv)

Merged genetic dataset for ML model training. **NO CLINICAL DATA** - purely genetics-based.

| Column | Type | Description | Source |
|--------|------|-------------|--------|
| sample_id | string | Unique identifier | Both |
| **Genetic Profile Features** | | | |
| ethnicity | string | Ethnicity (encoded) | genetic_profiles.csv |
| family_history_hearing_loss | integer | Family history flag | genetic_profiles.csv |
| consanguinity | integer | Consanguinity flag | genetic_profiles.csv |
| syndromic_genetic_condition | integer | Syndromic condition flag | genetic_profiles.csv |
| mtdna_variant_detected | integer | mtDNA variant flag | genetic_profiles.csv |
| polygenic_risk_score | float | Polygenic score | genetic_profiles.csv |
| gjb2_carrier | integer | GJB2 carrier status | genetic_profiles.csv |
| num_affected_relatives | integer | Affected relatives count | genetic_profiles.csv |
| **Variant Count Features** | | | |
| total_variant_count | integer | Total variants detected | variants.csv |
| pathogenic_variant_count | integer | Pathogenic variants | variants.csv |
| likely_pathogenic_count | integer | Likely pathogenic variants | variants.csv |
| vus_count | integer | VUS (uncertain) variants | variants.csv |
| benign_variant_count | integer | Benign variants | variants.csv |
| **Gene-Specific Features** | | | |
| has_gjb2_variant | integer | GJB2 variant present | variants.csv |
| has_slc26a4_variant | integer | SLC26A4 variant present | variants.csv |
| has_otof_variant | integer | OTOF variant present | variants.csv |
| has_myo7a_variant | integer | MYO7A variant present | variants.csv |
| has_cdh23_variant | integer | CDH23 variant present | variants.csv |
| has_tmc1_variant | integer | TMC1 variant present | variants.csv |
| **Zygosity & Impact Features** | | | |
| homozygous_variant_count | integer | Homozygous variants | variants.csv |
| compound_het_count | integer | Compound heterozygous | variants.csv |
| high_impact_count | integer | HIGH impact variants | variants.csv |
| moderate_impact_count | integer | MODERATE impact variants | variants.csv |
| **Pathogenicity Score Features** | | | |
| max_cadd_score | float | Maximum CADD score | variants.csv |
| mean_cadd_score | float | Mean CADD score | variants.csv |
| max_revel_score | float | Maximum REVEL score | variants.csv |
| rare_variant_count | integer | Rare variants (AF < 0.001) | variants.csv |
| unique_genes_affected | integer | Unique genes with variants | variants.csv |
| **Target Variables** | | | |
| genetic_risk_score | float | Calculated genetic risk (0-1) | Computed |
| hearing_impairment | integer | Binary outcome (0/1) | Computed |
| hearing_loss_severity | string | Severity category | Computed |

### Severity Categories (Based on Genetic Risk Score)
- **normal**: risk < 0.40
- **mild**: 0.40 ≤ risk < 0.50
- **moderate**: 0.50 ≤ risk < 0.65
- **severe**: 0.65 ≤ risk < 0.80
- **profound**: risk ≥ 0.80

---

## Key Differences from Clinical Approach

This genetics-only approach **excludes**:
- ❌ Age, sex demographics
- ❌ Birth weight, gestational age
- ❌ NICU/ventilation history
- ❌ Ototoxic medication exposure
- ❌ Maternal infections (CMV, rubella)
- ❌ Hyperbilirubinemia
- ❌ OAE/ABR screening results
- ❌ Audiometric thresholds

This genetics-only approach **focuses on**:
- ✅ Genomic variants in hearing loss genes
- ✅ Pathogenicity classifications
- ✅ Zygosity (homo/heterozygous/compound het)
- ✅ In-silico prediction scores (CADD, REVEL)
- ✅ Family history and inheritance patterns
- ✅ Polygenic risk assessment
