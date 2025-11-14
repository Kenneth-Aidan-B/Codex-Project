#!/usr/bin/env python3
"""
Preprocessing pipeline for hearing deficiency ML project.
Handles imputation, scaling, encoding, and class balancing (SMOTE).
"""
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "synthetic"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Random seed
SEED = 42
np.random.seed(SEED)


def load_data() -> pd.DataFrame:
    """Load the synthetic features dataset"""
    features_path = DATA_DIR / "features.csv"
    logger.info(f"Loading data from {features_path}")
    
    if not features_path.exists():
        logger.error(f"Features file not found: {features_path}")
        logger.error("Please run: python data/synthetic/generate_synthetic.py")
        sys.exit(1)
    
    df = pd.read_csv(features_path)
    logger.info(f"Loaded {len(df)} samples with {len(df.columns)} features")
    return df


def preprocess_features(df: pd.DataFrame) -> tuple:
    """
    Preprocess features: encoding, imputation, scaling
    Returns: X, y, feature_names, encoders
    """
    logger.info("Preprocessing features...")
    
    # Separate features and target
    target_col = 'hearing_impairment'
    id_col = 'sample_id'
    
    if target_col not in df.columns:
        logger.error(f"Target column '{target_col}' not found in dataset")
        sys.exit(1)
    
    # Store sample IDs
    sample_ids = df[id_col].copy()
    
    # Drop columns we don't need for training
    drop_cols = [id_col, target_col, 'hearing_loss_severity', 'diagnostic_abr_threshold_db']
    drop_cols = [col for col in drop_cols if col in df.columns]
    
    X = df.drop(columns=drop_cols)
    y = df[target_col]
    
    logger.info(f"Target distribution: {y.value_counts().to_dict()}")
    logger.info(f"Positive class rate: {y.mean():.2%}")
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    logger.info(f"Categorical features: {len(categorical_cols)}")
    logger.info(f"Numerical features: {len(numerical_cols)}")
    
    # Encode categorical features
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        # Handle missing values
        X[col] = X[col].fillna('unknown')
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
        logger.info(f"Encoded {col}: {len(le.classes_)} classes")
    
    # Impute missing numerical values
    imputer = SimpleImputer(strategy='median')
    if len(numerical_cols) > 0:
        X[numerical_cols] = imputer.fit_transform(X[numerical_cols])
        logger.info(f"Imputed missing values in {len(numerical_cols)} numerical features")
    
    # Scale numerical features
    scaler = StandardScaler()
    if len(numerical_cols) > 0:
        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
        logger.info("Scaled numerical features")
    
    feature_names = X.columns.tolist()
    
    return X.values, y.values, feature_names, sample_ids, {
        'label_encoders': label_encoders,
        'imputer': imputer,
        'scaler': scaler,
        'categorical_cols': categorical_cols,
        'numerical_cols': numerical_cols
    }


def apply_smote(X: np.ndarray, y: np.ndarray) -> tuple:
    """
    Apply SMOTE for class balancing
    """
    logger.info("Applying SMOTE for class balancing...")
    
    unique, counts = np.unique(y, return_counts=True)
    logger.info(f"Before SMOTE: {dict(zip(unique, counts))}")
    
    # SMOTE
    smote = SMOTE(random_state=SEED, k_neighbors=5)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    unique, counts = np.unique(y_resampled, return_counts=True)
    logger.info(f"After SMOTE: {dict(zip(unique, counts))}")
    
    return X_resampled, y_resampled


def save_processed_data(X: np.ndarray, y: np.ndarray, feature_names: list, 
                        sample_ids: pd.Series, encoders: dict) -> None:
    """Save processed data and preprocessing artifacts"""
    logger.info("Saving processed data...")
    
    # Save processed features as CSV
    df_processed = pd.DataFrame(X, columns=feature_names)
    df_processed['sample_id'] = sample_ids.values if len(sample_ids) == len(X) else ['synthetic'] * len(X)
    df_processed['hearing_impairment'] = y
    
    output_path = PROCESSED_DIR / "processed_features.csv"
    df_processed.to_csv(output_path, index=False)
    logger.info(f"✓ Saved processed features to {output_path}")
    
    # Save preprocessing artifacts
    artifacts = {
        'feature_names': feature_names,
        'encoders': encoders
    }
    
    artifacts_path = PROCESSED_DIR / "preprocessing_artifacts.joblib"
    joblib.dump(artifacts, artifacts_path)
    logger.info(f"✓ Saved preprocessing artifacts to {artifacts_path}")
    
    # Save metadata
    metadata = {
        'n_samples': len(X),
        'n_features': len(feature_names),
        'feature_names': feature_names,
        'categorical_features': encoders['categorical_cols'],
        'numerical_features': encoders['numerical_cols'],
        'target_distribution': {
            'negative': int((y == 0).sum()),
            'positive': int((y == 1).sum())
        }
    }
    
    metadata_path = PROCESSED_DIR / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"✓ Saved metadata to {metadata_path}")


def main():
    """Main preprocessing pipeline"""
    logger.info("="*60)
    logger.info("Starting preprocessing pipeline")
    logger.info("="*60)
    
    try:
        # Load data
        df = load_data()
        
        # Preprocess
        X, y, feature_names, sample_ids, encoders = preprocess_features(df)
        
        # Apply SMOTE
        X_balanced, y_balanced = apply_smote(X, y)
        
        # Save processed data
        save_processed_data(X_balanced, y_balanced, feature_names, sample_ids, encoders)
        
        # Create verification marker
        marker_path = PROCESSED_DIR / ".preprocess_ok"
        marker_path.touch()
        logger.info(f"✓ Created verification marker {marker_path}")
        
        logger.info("="*60)
        logger.info("✓ Preprocessing completed successfully!")
        logger.info("="*60)
        
        return 0
        
    except Exception as e:
        logger.error(f"✗ Error during preprocessing: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
