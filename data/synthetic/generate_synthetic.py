#!/usr/bin/env python3
"""
Synthetic data generator for GENETIC hearing deficiency ML project.
Generates purely genetics-based data for hearing loss prediction.
Focus: Genomic variants and genetic risk factors ONLY.
"""
import json
import logging
import sys
from pathlib import Path

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

# Hearing loss genes with their relative pathogenicity weights
HEARING_LOSS_GENES = {
    'GJB2': {'weight': 0.35, 'inheritance': 'recessive', 'chr': 'chr13'},
    'SLC26A4': {'weight': 0.25, 'inheritance': 'recessive', 'chr': 'chr7'},
    'OTOF': {'weight': 0.15, 'inheritance': 'recessive', 'chr': 'chr2'},
    'MYO7A': {'weight': 0.20, 'inheritance': 'recessive', 'chr': 'chr11'},
    'MYO15A': {'weight': 0.18, 'inheritance': 'recessive', 'chr': 'chr17'},
    'STRC': {'weight': 0.12, 'inheritance': 'recessive', 'chr': 'chr15'},
    'CDH23': {'weight': 0.22, 'inheritance': 'recessive', 'chr': 'chr10'},
    'PCDH15': {'weight': 0.20, 'inheritance': 'recessive', 'chr': 'chr10'},
    'TMC1': {'weight': 0.15, 'inheritance': 'recessive', 'chr': 'chr9'},
    'TMPRSS3': {'weight': 0.12, 'inheritance': 'recessive', 'chr': 'chr21'},
    'KCNQ4': {'weight': 0.10, 'inheritance': 'dominant', 'chr': 'chr1'},
    'TECTA': {'weight': 0.14, 'inheritance': 'both', 'chr': 'chr11'},
    'WFS1': {'weight': 0.12, 'inheritance': 'dominant', 'chr': 'chr4'},
    'TRIOBP': {'weight': 0.10, 'inheritance': 'recessive', 'chr': 'chr22'},
}

VARIANT_TYPES = ['SNV', 'INDEL', 'CNV']
ZYGOSITY = ['homozygous', 'heterozygous', 'compound_heterozygous']
PATHOGENICITY = ['pathogenic', 'likely_pathogenic', 'benign', 'likely_benign', 'VUS']
IMPACTS = ['HIGH', 'MODERATE', 'LOW', 'MODIFIER']
CONSEQUENCES = ['missense_variant', 'frameshift_variant', 'splice_site', 'stop_gained', 'synonymous_variant', 'intron_variant']
ETHNICITIES = ['Caucasian', 'Hispanic', 'Asian', 'African', 'Middle_Eastern', 'Mixed']


def generate_variants_data(num_samples: int) -> pd.DataFrame:
    """Generate synthetic genomic variants data"""
    logger.info(f"Generating genetic variants for {num_samples} samples...")
    
    records = []
    gene_list = list(HEARING_LOSS_GENES.keys())
    
    for i in range(num_samples):
        sample_id = f"SAMPLE_{i:04d}"
        num_variants = np.random.choice([0, 1, 2, 3, 4, 5, 6], p=[0.25, 0.35, 0.20, 0.12, 0.05, 0.02, 0.01])
        
        for v in range(num_variants):
            gene = np.random.choice(gene_list)
            gene_info = HEARING_LOSS_GENES[gene]
            chromosome = gene_info['chr']
            position = np.random.randint(10000000, 150000000)
            ref = np.random.choice(['A', 'C', 'G', 'T'])
            alt = np.random.choice([a for a in ['A', 'C', 'G', 'T'] if a != ref])
            
            variant_id = f"{chromosome}:{position}:{ref}>{alt}"
            variant_type = np.random.choice(VARIANT_TYPES, p=[0.75, 0.20, 0.05])
            
            if gene_info['inheritance'] == 'recessive':
                zygosity = np.random.choice(ZYGOSITY, p=[0.15, 0.70, 0.15])
            else:
                zygosity = np.random.choice(ZYGOSITY, p=[0.05, 0.90, 0.05])
            
            base_path_prob = gene_info['weight']
            if base_path_prob > 0.20:
                pathogenicity = np.random.choice(PATHOGENICITY, p=[0.30, 0.25, 0.15, 0.15, 0.15])
            else:
                pathogenicity = np.random.choice(PATHOGENICITY, p=[0.15, 0.20, 0.25, 0.25, 0.15])
            
            if pathogenicity in ['pathogenic', 'likely_pathogenic']:
                allele_freq = np.random.uniform(0.00001, 0.005)
                cadd_score = np.random.uniform(22, 40)
                revel_score = np.random.uniform(0.7, 1.0)
            elif pathogenicity == 'VUS':
                allele_freq = np.random.uniform(0.0001, 0.01)
                cadd_score = np.random.uniform(10, 25)
                revel_score = np.random.uniform(0.3, 0.7)
            else:
                allele_freq = np.random.uniform(0.001, 0.05)
                cadd_score = np.random.uniform(0, 18)
                revel_score = np.random.uniform(0.0, 0.4)
            
            consequence = np.random.choice(CONSEQUENCES)
            if consequence in ['frameshift_variant', 'stop_gained', 'splice_site']:
                impact = 'HIGH'
            elif consequence == 'missense_variant':
                impact = 'MODERATE'
            elif consequence == 'synonymous_variant':
                impact = 'LOW'
            else:
                impact = 'MODIFIER'
            
            records.append({
                'sample_id': sample_id, 'gene': gene, 'variant_id': variant_id,
                'chromosome': chromosome, 'position': position, 'ref': ref, 'alt': alt,
                'variant_type': variant_type, 'zygosity': zygosity,
                'allele_frequency': round(allele_freq, 8),
                'pathogenicity': pathogenicity, 'cadd_score': round(cadd_score, 2),
                'revel_score': round(revel_score, 3), 'impact': impact, 'consequence': consequence,
                'inheritance_pattern': gene_info['inheritance']
            })
    
    df = pd.DataFrame(records)
    logger.info(f"Generated {len(df)} variant records")
    return df


def generate_genetic_profiles(num_samples: int) -> pd.DataFrame:
    """Generate genetic profile data"""
    logger.info(f"Generating genetic profiles for {num_samples} samples...")
    
    records = []
    for i in range(num_samples):
        sample_id = f"SAMPLE_{i:04d}"
        ethnicity = np.random.choice(ETHNICITIES, p=[0.30, 0.20, 0.20, 0.15, 0.05, 0.10])
        family_history = np.random.choice([0, 1], p=[0.80, 0.20])
        consanguinity = np.random.choice([0, 1], p=[0.92, 0.08])
        syndromic_condition = np.random.choice([0, 1], p=[0.92, 0.08])
        mtdna_variant = np.random.choice([0, 1], p=[0.95, 0.05])
        polygenic_score = round(np.clip(np.random.normal(0.5, 0.15), 0, 1), 4)
        gjb2_carrier = np.random.choice([0, 1], p=[0.97, 0.03])
        num_affected_relatives = np.random.choice([0, 1, 2, 3], p=[0.75, 0.15, 0.07, 0.03]) if family_history else 0
        
        records.append({
            'sample_id': sample_id, 'ethnicity': ethnicity,
            'family_history_hearing_loss': family_history, 'consanguinity': consanguinity,
            'syndromic_genetic_condition': syndromic_condition,
            'mtdna_variant_detected': mtdna_variant, 'polygenic_risk_score': polygenic_score,
            'gjb2_carrier': gjb2_carrier, 'num_affected_relatives': num_affected_relatives
        })
    
    return pd.DataFrame(records)


def create_features_dataset(variants_df: pd.DataFrame, profiles_df: pd.DataFrame) -> pd.DataFrame:
    """Create genetic-only features dataset"""
    logger.info("Creating genetic features dataset...")
    
    variant_features = []
    for sample_id in profiles_df['sample_id']:
        sv = variants_df[variants_df['sample_id'] == sample_id]
        
        features = {
            'sample_id': sample_id,
            'total_variant_count': len(sv),
            'pathogenic_variant_count': len(sv[sv['pathogenicity'] == 'pathogenic']),
            'likely_pathogenic_count': len(sv[sv['pathogenicity'] == 'likely_pathogenic']),
            'vus_count': len(sv[sv['pathogenicity'] == 'VUS']),
            'benign_variant_count': len(sv[sv['pathogenicity'].isin(['benign', 'likely_benign'])]),
            'has_gjb2_variant': int('GJB2' in sv['gene'].values),
            'has_slc26a4_variant': int('SLC26A4' in sv['gene'].values),
            'has_otof_variant': int('OTOF' in sv['gene'].values),
            'has_myo7a_variant': int('MYO7A' in sv['gene'].values),
            'has_cdh23_variant': int('CDH23' in sv['gene'].values),
            'has_tmc1_variant': int('TMC1' in sv['gene'].values),
            'homozygous_variant_count': len(sv[sv['zygosity'] == 'homozygous']),
            'compound_het_count': len(sv[sv['zygosity'] == 'compound_heterozygous']),
            'high_impact_count': len(sv[sv['impact'] == 'HIGH']),
            'moderate_impact_count': len(sv[sv['impact'] == 'MODERATE']),
            'max_cadd_score': sv['cadd_score'].max() if len(sv) > 0 else 0.0,
            'mean_cadd_score': sv['cadd_score'].mean() if len(sv) > 0 else 0.0,
            'max_revel_score': sv['revel_score'].max() if len(sv) > 0 else 0.0,
            'rare_variant_count': len(sv[sv['allele_frequency'] < 0.001]),
            'unique_genes_affected': sv['gene'].nunique() if len(sv) > 0 else 0
        }
        variant_features.append(features)
    
    variant_df = pd.DataFrame(variant_features)
    features_df = profiles_df.merge(variant_df, on='sample_id', how='left').fillna(0)
    
    # Calculate genetic risk
    risk = pd.Series(0.0, index=features_df.index)
    risk += features_df['pathogenic_variant_count'] * 0.25
    risk += features_df['likely_pathogenic_count'] * 0.15
    risk += features_df['homozygous_variant_count'] * 0.20
    risk += features_df['compound_het_count'] * 0.18
    risk += features_df['high_impact_count'] * 0.12
    risk += features_df['has_gjb2_variant'] * 0.15
    risk += features_df['has_slc26a4_variant'] * 0.10
    risk += features_df['has_myo7a_variant'] * 0.12
    risk += features_df['has_cdh23_variant'] * 0.12
    risk += (features_df['max_cadd_score'] / 40) * 0.10
    risk += features_df['max_revel_score'] * 0.08
    risk += features_df['family_history_hearing_loss'] * 0.15
    risk += features_df['consanguinity'] * 0.12
    risk += features_df['syndromic_genetic_condition'] * 0.20
    risk += features_df['mtdna_variant_detected'] * 0.15
    risk += features_df['num_affected_relatives'] * 0.05
    risk += np.random.normal(0, 0.05, len(features_df))
    
    features_df['genetic_risk_score'] = np.clip(risk, 0, 1).round(4)
    features_df['hearing_impairment'] = (features_df['genetic_risk_score'] > 0.40).astype(int)
    features_df['hearing_loss_severity'] = features_df['genetic_risk_score'].apply(
        lambda x: 'normal' if x < 0.40 else 'mild' if x < 0.50 else 'moderate' if x < 0.65 else 'severe' if x < 0.80 else 'profound'
    )
    
    logger.info(f"Created {len(features_df)} samples, {len(features_df.columns)} features")
    logger.info(f"Hearing impairment rate: {features_df['hearing_impairment'].mean():.2%}")
    return features_df


def main():
    logger.info("="*60)
    logger.info("GENETIC-ONLY Synthetic Data Generation")
    logger.info("="*60)
    
    try:
        variants_df = generate_variants_data(NUM_SAMPLES)
        profiles_df = generate_genetic_profiles(NUM_SAMPLES)
        features_df = create_features_dataset(variants_df, profiles_df)
        
        variants_df.to_csv(DATA_DIR / "variants.csv", index=False)
        profiles_df.to_csv(DATA_DIR / "genetic_profiles.csv", index=False)
        features_df.to_csv(DATA_DIR / "features.csv", index=False)
        
        # Remove old clinical.csv
        old_clinical = DATA_DIR / "clinical.csv"
        if old_clinical.exists():
            old_clinical.unlink()
        
        summary = {
            'num_samples': NUM_SAMPLES, 'num_variants': len(variants_df),
            'hearing_impairment_rate': float(features_df['hearing_impairment'].mean()),
            'seed': SEED, 'data_type': 'genetic_only',
            'severity_distribution': features_df['hearing_loss_severity'].value_counts().to_dict(),
            'features': list(features_df.columns)
        }
        with open(DATA_DIR / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("✓ Genetic-only data generation complete!")
        return 0
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
