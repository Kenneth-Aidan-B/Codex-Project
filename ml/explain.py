#!/usr/bin/env python3
"""
SHAP explainability for hearing deficiency ML models.
Generates per-sample and global explanations.
"""
import json
import logging
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
EXPLANATIONS_DIR = RESULTS_DIR / "explanations"
EXPLANATIONS_DIR.mkdir(parents=True, exist_ok=True)

# Random seed
SEED = 42
np.random.seed(SEED)


def load_processed_data() -> tuple:
    """Load preprocessed data"""
    features_path = PROCESSED_DIR / "processed_features.csv"
    logger.info(f"Loading processed data from {features_path}")
    
    if not features_path.exists():
        logger.error(f"Processed data not found: {features_path}")
        sys.exit(1)
    
    df = pd.read_csv(features_path)
    
    # Separate features and target
    sample_ids = df['sample_id'].values
    X = df.drop(columns=['sample_id', 'hearing_impairment'])
    y = df['hearing_impairment']
    feature_names = X.columns.tolist()
    
    logger.info(f"Loaded {len(X)} samples with {len(feature_names)} features")
    
    return X.values, y.values, feature_names, sample_ids


def load_model(model_name: str):
    """Load a trained model"""
    model_dir = MODELS_DIR / model_name
    model_path = model_dir / "model_latest.joblib"
    
    if not model_path.exists():
        logger.warning(f"Model not found: {model_path}")
        return None
    
    try:
        model = joblib.load(model_path)
        logger.info(f"✓ Loaded {model_name}")
        return model
    except Exception as e:
        logger.error(f"Error loading {model_name}: {e}")
        return None


def explain_tree_model(model, X, feature_names, model_name, sample_ids, n_samples=100):
    """Generate SHAP explanations for tree-based models"""
    logger.info(f"Generating SHAP explanations for {model_name}...")
    
    try:
        # Use a subset for speed
        X_explain = X[:n_samples]
        sample_ids_explain = sample_ids[:n_samples]
        
        # Create TreeExplainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_explain)
        
        # For binary classification, some models return 2D array
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class
        elif len(shap_values.shape) > 2:
            shap_values = shap_values[:, :, 1]  # Use positive class
        
        logger.info(f"✓ Generated SHAP values for {model_name}")
        
        # Generate per-sample explanations
        for i, sample_id in enumerate(sample_ids_explain):
            explanation = {
                'sample_id': str(sample_id),
                'model': model_name,
                'shap_values': {
                    feature_names[j]: float(shap_values[i, j])
                    for j in range(len(feature_names))
                },
                'top_features': sorted(
                    [(feature_names[j], float(shap_values[i, j])) 
                     for j in range(len(feature_names))],
                    key=lambda x: abs(x[1]),
                    reverse=True
                )[:10]
            }
            
            explanation_path = EXPLANATIONS_DIR / f"{sample_id}_{model_name}.json"
            with open(explanation_path, 'w') as f:
                json.dump(explanation, f, indent=2)
        
        logger.info(f"✓ Saved {len(sample_ids_explain)} sample explanations for {model_name}")
        
        # Global feature importance
        feature_importance = np.abs(shap_values).mean(axis=0)
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance,
            'model': model_name
        }).sort_values('importance', ascending=False)
        
        return importance_df, shap_values
        
    except Exception as e:
        logger.error(f"Error explaining {model_name}: {e}")
        return None, None


def explain_with_kernel(model, X, feature_names, model_name, sample_ids, n_samples=50, n_background=50):
    """Generate SHAP explanations using KernelExplainer (for non-tree models)"""
    logger.info(f"Generating SHAP explanations for {model_name} using KernelExplainer...")
    
    try:
        # Use smaller subsets for KernelExplainer (it's slow)
        X_background = X[:n_background]
        X_explain = X[:n_samples]
        sample_ids_explain = sample_ids[:n_samples]
        
        # Create KernelExplainer
        explainer = shap.KernelExplainer(model.predict_proba, X_background)
        shap_values = explainer.shap_values(X_explain)
        
        # For binary classification
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class
        
        logger.info(f"✓ Generated SHAP values for {model_name}")
        
        # Generate per-sample explanations (limited)
        for i, sample_id in enumerate(sample_ids_explain[:20]):  # Only first 20 for speed
            explanation = {
                'sample_id': str(sample_id),
                'model': model_name,
                'shap_values': {
                    feature_names[j]: float(shap_values[i, j])
                    for j in range(len(feature_names))
                },
                'top_features': sorted(
                    [(feature_names[j], float(shap_values[i, j])) 
                     for j in range(len(feature_names))],
                    key=lambda x: abs(x[1]),
                    reverse=True
                )[:10]
            }
            
            explanation_path = EXPLANATIONS_DIR / f"{sample_id}_{model_name}.json"
            with open(explanation_path, 'w') as f:
                json.dump(explanation, f, indent=2)
        
        logger.info(f"✓ Saved sample explanations for {model_name}")
        
        # Global feature importance
        feature_importance = np.abs(shap_values).mean(axis=0)
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance,
            'model': model_name
        }).sort_values('importance', ascending=False)
        
        return importance_df, shap_values
        
    except Exception as e:
        logger.error(f"Error explaining {model_name}: {e}")
        return None, None


def save_global_importance(all_importance_dfs):
    """Save global feature importance across all models"""
    if not all_importance_dfs:
        logger.warning("No importance dataframes to save")
        return
    
    combined_df = pd.concat(all_importance_dfs, ignore_index=True)
    
    # Pivot to have models as columns
    pivot_df = combined_df.pivot(index='feature', columns='model', values='importance')
    pivot_df['mean_importance'] = pivot_df.mean(axis=1)
    pivot_df = pivot_df.sort_values('mean_importance', ascending=False)
    
    output_path = RESULTS_DIR / "feature_importance.csv"
    pivot_df.to_csv(output_path)
    logger.info(f"✓ Saved global feature importance to {output_path}")
    
    # Also save top features
    top_features = pivot_df.head(20)
    logger.info(f"\nTop 20 most important features:")
    for idx, (feature, row) in enumerate(top_features.iterrows(), 1):
        logger.info(f"  {idx}. {feature}: {row['mean_importance']:.4f}")


def main():
    """Main explanation pipeline"""
    logger.info("="*60)
    logger.info("Starting SHAP explanation pipeline")
    logger.info("="*60)
    
    try:
        # Load data
        X, y, feature_names, sample_ids = load_processed_data()
        
        all_importance_dfs = []
        
        # Explain tree-based models (faster)
        tree_models = ['RandomForest', 'XGBoost']
        for model_name in tree_models:
            model = load_model(model_name)
            if model is not None:
                importance_df, shap_values = explain_tree_model(
                    model, X, feature_names, model_name, sample_ids, n_samples=100
                )
                if importance_df is not None:
                    all_importance_dfs.append(importance_df)
        
        # Explain SVM (slower with KernelExplainer)
        svm_model = load_model('SVM')
        if svm_model is not None:
            importance_df, shap_values = explain_with_kernel(
                svm_model, X, feature_names, 'SVM', sample_ids, n_samples=30, n_background=30
            )
            if importance_df is not None:
                all_importance_dfs.append(importance_df)
        
        # Save global importance
        save_global_importance(all_importance_dfs)
        
        # Create verification marker
        marker_path = RESULTS_DIR / ".explain_ok"
        marker_path.touch()
        logger.info(f"✓ Created verification marker {marker_path}")
        
        logger.info("="*60)
        logger.info("✓ SHAP explanation completed successfully!")
        logger.info("="*60)
        
        return 0
        
    except Exception as e:
        logger.error(f"✗ Error during explanation: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
