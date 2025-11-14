#!/usr/bin/env python3
"""
Synthetic data generator for hearing deficiency ML project.
Generates realistic genomic variants, clinical data, and merged features.
"""
import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
SEED = 42
NUM_SAMPLES = 1000
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "synthetic"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Seed for reproducibility
np.random.seed(SEED)

# Hearing loss genes
HEARING_LOSS_GENES = [
    'GJB2', 'SLC26A4', 'OTOF', 'MYO7A', 'MYO15A', 
    'STRC', 'CDH23', 'PCDH15', 'TMC1', 'TMPRSS3',
    'KCNQ4', 'TECTA', 'COL11A2', 'MYO6', 'ESPN'
]

CHROMOSOMES = ['chr1', 'chr7', 'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr17', 'chrX']
VARIANT_TYPES = ['SNV', 'INDEL', 'CNV']
ZYGOSITY = ['homozygous', 'heterozygous', 'compound_heterozygous']
PATHOGENICITY = ['pathogenic', 'likely_pathogenic', 'benign', 'likely_benign', 'VUS']
IMPACTS = ['HIGH', 'MODERATE', 'LOW', 'MODIFIER']
CONSEQUENCES = ['missense_variant', 'frameshift', 'splice_site', 'nonsense', 'synonymous', 'intron_variant']
ETHNICITIES = ['Caucasian', 'Hispanic', 'Asian', 'African', 'Mixed']
OAE_RESULTS = ['pass', 'refer', 'not_done']
AABR_RESULTS = ['pass', 'refer', 'not_done']
SEVERITY_CLASSES = ['normal', 'mild', 'moderate', 'severe', 'profound']


def generate_variants_data(num_samples: int) -> pd.DataFrame:
    """Generate synthetic genomic variants data"""
    logger.info(f"Generating variants data for {num_samples} samples...")
    
    records = []
    for i in range(num_samples):
        sample_id = f"SAMPLE_{i:04d}"
        
        # Each sample has 0-5 variants
        num_variants = np.random.choice([0, 1, 2, 3, 4, 5], p=[0.3, 0.35, 0.2, 0.1, 0.04, 0.01])
        
        for v in range(num_variants):
            gene = np.random.choice(HEARING_LOSS_GENES)
            chromosome = np.random.choice(CHROMOSOMES)
            position = np.random.randint(1000000, 50000000)
            ref = np.random.choice(['A', 'C', 'G', 'T'])
            alt = np.random.choice([a for a in ['A', 'C', 'G', 'T'] if a != ref])
            
            variant_id = f"{chromosome}:{position}:{ref}>{alt}"
            variant_type = np.random.choice(VARIANT_TYPES, p=[0.8, 0.15, 0.05])
            zygosity = np.random.choice(ZYGOSITY, p=[0.1, 0.8, 0.1])
            
            # Pathogenicity: GJB2 and SLC26A4 more likely pathogenic
            if gene in ['GJB2', 'SLC26A4']:
                pathogenicity = np.random.choice(PATHOGENICITY, p=[0.3, 0.25, 0.15, 0.15, 0.15])
            else:
                pathogenicity = np.random.choice(PATHOGENICITY, p=[0.1, 0.15, 0.3, 0.25, 0.2])
            
            # Allele frequency: pathogenic variants are rarer
            if pathogenicity in ['pathogenic', 'likely_pathogenic']:
                allele_freq = np.random.uniform(0.0001, 0.01)
            else:
                allele_freq = np.random.uniform(0.001, 0.05)
            
            # CADD score: higher for pathogenic
            if pathogenicity in ['pathogenic', 'likely_pathogenic']:
                cadd_score = np.random.uniform(20, 35)
            else:
                cadd_score = np.random.uniform(5, 25)
            
            impact = np.random.choice(IMPACTS, p=[0.15, 0.35, 0.35, 0.15])
            consequence = np.random.choice(CONSEQUENCES)
            
            records.append({
                'sample_id': sample_id,
                'gene': gene,
                'variant_id': variant_id,
                'chromosome': chromosome,
                'position': position,
                'ref': ref,
                'alt': alt,
                'variant_type': variant_type,
                'zygosity': zygosity,
                'allele_frequency': allele_freq,
                'pathogenicity': pathogenicity,
                'cadd_score': cadd_score,
                'impact': impact,
                'consequence': consequence
            })
    
    df = pd.DataFrame(records)
    logger.info(f"Generated {len(df)} variant records")
    return df


def generate_clinical_data(num_samples: int) -> pd.DataFrame:
    """Generate synthetic clinical data"""
    logger.info(f"Generating clinical data for {num_samples} samples...")
    
    records = []
    for i in range(num_samples):
        sample_id = f"SAMPLE_{i:04d}"
        
        # Demographics
        age_months = np.random.randint(0, 37)
        sex = np.random.choice(['M', 'F'])
        ethnicity = np.random.choice(ETHNICITIES)
        
        # Birth characteristics
        gestational_age = np.clip(np.random.normal(38.5, 2.5), 24, 42)
        premature = 1 if gestational_age < 37 else 0
        birth_weight = int(np.clip(np.random.normal(3200, 600), 500, 5000))
        
        # APGAR scores (skewed toward higher values)
        apgar_1_probs = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.12, 0.20, 0.25, 0.15, 0.05])
        apgar_1min = np.random.choice(range(11), p=apgar_1_probs / apgar_1_probs.sum())
        apgar_5_probs = np.array([0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08, 0.15, 0.30, 0.26, 0.075])
        apgar_5min = np.random.choice(range(11), p=apgar_5_probs / apgar_5_probs.sum())
        
        # NICU and complications
        nicu_days = np.random.choice([0] * 70 + list(range(1, 121)), size=1)[0]
        mech_vent_days = min(nicu_days, np.random.randint(0, 61)) if nicu_days > 0 else 0
        
        # Hyperbilirubinemia
        hyperbilirubinemia = np.random.choice([0, 1], p=[0.85, 0.15])
        bilirubin_max = np.random.uniform(5, 25) if hyperbilirubinemia else np.random.uniform(1, 8)
        
        # Ototoxic medications
        ototoxic_meds = np.random.choice([0, 1], p=[0.90, 0.10])
        aminoglycoside_days = np.random.randint(0, 31) if ototoxic_meds else 0
        loop_diuretic_days = np.random.randint(0, 31) if ototoxic_meds else 0
        
        # Maternal infections
        maternal_cmv = np.random.choice([0, 1], p=[0.98, 0.02])
        maternal_rubella = np.random.choice([0, 1], p=[0.99, 0.01])
        maternal_toxo = np.random.choice([0, 1], p=[0.99, 0.01])
        
        # Family and syndromic features
        family_history = np.random.choice([0, 1], p=[0.85, 0.15])
        consanguinity = np.random.choice([0, 1], p=[0.95, 0.05])
        syndromic = np.random.choice([0, 1], p=[0.90, 0.10])
        craniofacial = np.random.choice([0, 1], p=[0.92, 0.08])
        
        # Screening results
        oae_result = np.random.choice(OAE_RESULTS, p=[0.85, 0.12, 0.03])
        aabr_result = np.random.choice(AABR_RESULTS, p=[0.88, 0.10, 0.02])
        
        # Determine hearing loss (ground truth based on risk factors)
        risk_score = 0.0
        
        # Strong risk factors
        if family_history: risk_score += 0.15
        if premature: risk_score += 0.10
        if nicu_days > 5: risk_score += 0.12
        if mech_vent_days > 0: risk_score += 0.08
        if hyperbilirubinemia and bilirubin_max > 20: risk_score += 0.15
        if ototoxic_meds and aminoglycoside_days > 7: risk_score += 0.12
        if maternal_cmv: risk_score += 0.20
        if syndromic: risk_score += 0.18
        if craniofacial: risk_score += 0.10
        if apgar_5min < 6: risk_score += 0.10
        if oae_result == 'refer': risk_score += 0.25
        if aabr_result == 'refer': risk_score += 0.30
        
        # Add randomness
        risk_score = np.clip(risk_score + np.random.normal(0, 0.1), 0, 1)
        
        # Determine hearing loss (15-20% prevalence)
        hearing_impairment = 1 if risk_score > 0.35 else 0
        
        # Severity based on risk score
        if hearing_impairment:
            if risk_score < 0.45:
                severity = 'mild'
                abr_threshold = np.random.uniform(25, 40)
            elif risk_score < 0.60:
                severity = 'moderate'
                abr_threshold = np.random.uniform(41, 70)
            elif risk_score < 0.75:
                severity = 'severe'
                abr_threshold = np.random.uniform(71, 90)
            else:
                severity = 'profound'
                abr_threshold = np.random.uniform(91, 100)
        else:
            severity = 'normal'
            abr_threshold = np.random.uniform(10, 25)
        
        records.append({
            'sample_id': sample_id,
            'age_months': age_months,
            'sex': sex,
            'ethnicity': ethnicity,
            'birth_weight_g': birth_weight,
            'gestational_age_weeks': round(gestational_age, 1),
            'premature': premature,
            'apgar_1min': apgar_1min,
            'apgar_5min': apgar_5min,
            'nicu_days': nicu_days,
            'mechanical_ventilation_days': mech_vent_days,
            'hyperbilirubinemia': hyperbilirubinemia,
            'bilirubin_max_mg_dl': round(bilirubin_max, 2),
            'ototoxic_medications': ototoxic_meds,
            'aminoglycoside_days': aminoglycoside_days,
            'loop_diuretic_days': loop_diuretic_days,
            'maternal_cmv_infection': maternal_cmv,
            'maternal_rubella': maternal_rubella,
            'maternal_toxoplasmosis': maternal_toxo,
            'family_history_hearing_loss': family_history,
            'consanguinity': consanguinity,
            'syndromic_features': syndromic,
            'craniofacial_anomalies': craniofacial,
            'oae_result': oae_result,
            'aabr_result': aabr_result,
            'diagnostic_abr_threshold_db': round(abr_threshold, 1),
            'hearing_loss_severity': severity,
            'hearing_impairment': hearing_impairment
        })
    
    df = pd.DataFrame(records)
    logger.info(f"Generated {len(df)} clinical records")
    logger.info(f"Hearing impairment prevalence: {df['hearing_impairment'].mean():.2%}")
    return df


def create_features_dataset(variants_df: pd.DataFrame, clinical_df: pd.DataFrame) -> pd.DataFrame:
    """Merge variants and clinical data into features dataset"""
    logger.info("Creating merged features dataset...")
    
    # Aggregate variant features per sample
    variant_features = []
    
    for sample_id in clinical_df['sample_id']:
        sample_variants = variants_df[variants_df['sample_id'] == sample_id]
        
        features = {
            'sample_id': sample_id,
            'pathogenic_variant_count': len(sample_variants[
                sample_variants['pathogenicity'].isin(['pathogenic', 'likely_pathogenic'])
            ]),
            'has_gjb2_variant': int('GJB2' in sample_variants['gene'].values),
            'has_slc26a4_variant': int('SLC26A4' in sample_variants['gene'].values),
            'has_otof_variant': int('OTOF' in sample_variants['gene'].values),
            'max_cadd_score': sample_variants['cadd_score'].max() if len(sample_variants) > 0 else 0.0,
            'homozygous_variant_count': len(sample_variants[sample_variants['zygosity'] == 'homozygous']),
            'compound_het_count': len(sample_variants[sample_variants['zygosity'] == 'compound_heterozygous']),
            'high_impact_variant_count': len(sample_variants[sample_variants['impact'] == 'HIGH'])
        }
        variant_features.append(features)
    
    variant_features_df = pd.DataFrame(variant_features)
    
    # Merge with clinical data
    features_df = clinical_df.merge(variant_features_df, on='sample_id', how='left')
    features_df = features_df.fillna(0)
    
    logger.info(f"Created features dataset with {len(features_df)} samples and {len(features_df.columns)} features")
    return features_df


def main():
    """Main execution function"""
    logger.info("="*60)
    logger.info("Starting synthetic data generation")
    logger.info(f"Seed: {SEED}, Samples: {NUM_SAMPLES}")
    logger.info("="*60)
    
    try:
        # Generate data
        variants_df = generate_variants_data(NUM_SAMPLES)
        clinical_df = generate_clinical_data(NUM_SAMPLES)
        features_df = create_features_dataset(variants_df, clinical_df)
        
        # Save to CSV files
        variants_path = DATA_DIR / "variants.csv"
        clinical_path = DATA_DIR / "clinical.csv"
        features_path = DATA_DIR / "features.csv"
        
        variants_df.to_csv(variants_path, index=False)
        clinical_df.to_csv(clinical_path, index=False)
        features_df.to_csv(features_path, index=False)
        
        logger.info(f"✓ Saved variants to {variants_path}")
        logger.info(f"✓ Saved clinical data to {clinical_path}")
        logger.info(f"✓ Saved features to {features_path}")
        
        # Save summary statistics
        summary = {
            'num_samples': NUM_SAMPLES,
            'num_variants': len(variants_df),
            'hearing_impairment_rate': float(features_df['hearing_impairment'].mean()),
            'seed': SEED,
            'severity_distribution': features_df['hearing_loss_severity'].value_counts().to_dict()
        }
        
        summary_path = DATA_DIR / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"✓ Saved summary to {summary_path}")
        
        # Create verification marker
        marker_path = DATA_DIR / ".generated_ok"
        marker_path.touch()
        logger.info(f"✓ Created verification marker {marker_path}")
        
        logger.info("="*60)
        logger.info("✓ Synthetic data generation completed successfully!")
        logger.info("="*60)
        
        return 0
        
    except Exception as e:
        logger.error(f"✗ Error during data generation: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
