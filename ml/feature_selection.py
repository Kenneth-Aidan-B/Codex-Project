#!/usr/bin/env python3
"""
Feature selection for hearing deficiency ML project.
Implements ReliefF approximation, PCA, and mutual information.
"""
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.neighbors import NearestNeighbors

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Random seed
SEED = 42
np.random.seed(SEED)


def load_processed_data() -> tuple:
    """Load preprocessed data"""
    features_path = PROCESSED_DIR / "processed_features.csv"
    logger.info(f"Loading processed data from {features_path}")
    
    if not features_path.exists():
        logger.error(f"Processed data not found: {features_path}")
        logger.error("Please run: python ml/preprocess.py")
        sys.exit(1)
    
    df = pd.read_csv(features_path)
    
    # Separate features and target
    X = df.drop(columns=['sample_id', 'hearing_impairment'])
    y = df['hearing_impairment']
    feature_names = X.columns.tolist()
    
    logger.info(f"Loaded {len(X)} samples with {len(feature_names)} features")
    
    return X.values, y.values, feature_names


def relieff_approximation(X: np.ndarray, y: np.ndarray, feature_names: list, k: int = 5) -> pd.DataFrame:
    """
    Simple ReliefF-like feature ranking
    For each feature, compute average difference with same-class and different-class nearest neighbors
    """
    logger.info(f"Computing ReliefF-like scores (k={k})...")
    
    n_samples, n_features = X.shape
    scores = np.zeros(n_features)
    
    # Find nearest neighbors
    nn = NearestNeighbors(n_neighbors=k+1, metric='euclidean')
    nn.fit(X)
    
    # For efficiency, sample a subset if dataset is large
    sample_size = min(200, n_samples)
    sample_indices = np.random.choice(n_samples, sample_size, replace=False)
    
    for idx in sample_indices:
        x_i = X[idx:idx+1]
        y_i = y[idx]
        
        # Find k+1 nearest neighbors (including itself)
        distances, indices = nn.kneighbors(x_i)
        neighbor_indices = indices[0][1:]  # Exclude itself
        neighbor_labels = y[neighbor_indices]
        
        # Separate same-class and different-class neighbors
        same_class = neighbor_indices[neighbor_labels == y_i]
        diff_class = neighbor_indices[neighbor_labels != y_i]
        
        if len(same_class) > 0 and len(diff_class) > 0:
            for f in range(n_features):
                # Distance to same-class neighbors (we want this to be small)
                same_dist = np.mean(np.abs(X[same_class, f] - X[idx, f]))
                # Distance to different-class neighbors (we want this to be large)
                diff_dist = np.mean(np.abs(X[diff_class, f] - X[idx, f]))
                # ReliefF-like score: higher is better
                scores[f] += (diff_dist - same_dist)
    
    scores /= sample_size
    
    # Create DataFrame with scores
    df_scores = pd.DataFrame({
        'feature': feature_names,
        'relieff_score': scores
    })
    df_scores = df_scores.sort_values('relieff_score', ascending=False)
    
    logger.info(f"✓ Computed ReliefF scores for {n_features} features")
    logger.info(f"Top 5 features: {df_scores['feature'].head(5).tolist()}")
    
    return df_scores


def compute_mutual_information(X: np.ndarray, y: np.ndarray, feature_names: list) -> pd.DataFrame:
    """Compute mutual information scores"""
    logger.info("Computing mutual information scores...")
    
    mi_scores = mutual_info_classif(X, y, random_state=SEED)
    
    df_scores = pd.DataFrame({
        'feature': feature_names,
        'mutual_info_score': mi_scores
    })
    df_scores = df_scores.sort_values('mutual_info_score', ascending=False)
    
    logger.info(f"✓ Computed MI scores for {len(feature_names)} features")
    logger.info(f"Top 5 features: {df_scores['feature'].head(5).tolist()}")
    
    return df_scores


def compute_pca(X: np.ndarray, feature_names: list, n_components: int = 10) -> dict:
    """Compute PCA for dimensionality reduction"""
    logger.info(f"Computing PCA with {n_components} components...")
    
    n_components = min(n_components, X.shape[1], X.shape[0])
    
    pca = PCA(n_components=n_components, random_state=SEED)
    X_pca = pca.fit_transform(X)
    
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    logger.info(f"✓ Computed PCA")
    logger.info(f"Explained variance by top 5 components: {explained_variance[:5]}")
    logger.info(f"Cumulative variance (first {n_components} components): {cumulative_variance[-1]:.2%}")
    
    # Get feature importance from PCA components
    component_importance = np.abs(pca.components_).sum(axis=0)
    df_pca_importance = pd.DataFrame({
        'feature': feature_names,
        'pca_importance': component_importance
    })
    df_pca_importance = df_pca_importance.sort_values('pca_importance', ascending=False)
    
    return {
        'pca_model': pca,
        'explained_variance': explained_variance,
        'cumulative_variance': cumulative_variance,
        'importance_df': df_pca_importance
    }


def select_features(relieff_df: pd.DataFrame, mi_df: pd.DataFrame, 
                   pca_dict: dict, top_k: int = 20) -> list:
    """
    Select top features based on combined rankings
    """
    logger.info(f"Selecting top {top_k} features...")
    
    # Normalize scores to 0-1 range
    relieff_df['relieff_norm'] = (relieff_df['relieff_score'] - relieff_df['relieff_score'].min()) / \
                                  (relieff_df['relieff_score'].max() - relieff_df['relieff_score'].min() + 1e-10)
    
    mi_df['mi_norm'] = (mi_df['mutual_info_score'] - mi_df['mutual_info_score'].min()) / \
                       (mi_df['mutual_info_score'].max() - mi_df['mutual_info_score'].min() + 1e-10)
    
    pca_df = pca_dict['importance_df']
    pca_df['pca_norm'] = (pca_df['pca_importance'] - pca_df['pca_importance'].min()) / \
                         (pca_df['pca_importance'].max() - pca_df['pca_importance'].min() + 1e-10)
    
    # Merge all scores
    combined = relieff_df[['feature', 'relieff_norm']].merge(
        mi_df[['feature', 'mi_norm']], on='feature'
    ).merge(
        pca_df[['feature', 'pca_norm']], on='feature'
    )
    
    # Compute combined score (equal weighting)
    combined['combined_score'] = (combined['relieff_norm'] + 
                                  combined['mi_norm'] + 
                                  combined['pca_norm']) / 3
    
    combined = combined.sort_values('combined_score', ascending=False)
    
    # Select top k features
    selected_features = combined['feature'].head(top_k).tolist()
    
    logger.info(f"✓ Selected {len(selected_features)} features")
    logger.info(f"Top 10 selected: {selected_features[:10]}")
    
    return selected_features, combined


def save_results(relieff_df: pd.DataFrame, mi_df: pd.DataFrame, pca_dict: dict, 
                selected_features: list, combined_scores: pd.DataFrame) -> None:
    """Save feature selection results"""
    logger.info("Saving feature selection results...")
    
    # Save ReliefF scores
    relieff_path = RESULTS_DIR / "relieff_scores.csv"
    relieff_df.to_csv(relieff_path, index=False)
    logger.info(f"✓ Saved ReliefF scores to {relieff_path}")
    
    # Save MI scores
    mi_path = RESULTS_DIR / "mutual_info_scores.csv"
    mi_df.to_csv(mi_path, index=False)
    logger.info(f"✓ Saved MI scores to {mi_path}")
    
    # Save PCA results
    pca_path = RESULTS_DIR / "pca_importance.csv"
    pca_dict['importance_df'].to_csv(pca_path, index=False)
    logger.info(f"✓ Saved PCA importance to {pca_path}")
    
    # Save PCA model
    pca_model_path = RESULTS_DIR / "pca_model.joblib"
    joblib.dump(pca_dict['pca_model'], pca_model_path)
    logger.info(f"✓ Saved PCA model to {pca_model_path}")
    
    # Save combined scores
    combined_path = RESULTS_DIR / "combined_feature_scores.csv"
    combined_scores.to_csv(combined_path, index=False)
    logger.info(f"✓ Saved combined scores to {combined_path}")
    
    # Save selected features
    selected_features_dict = {
        'selected_features': selected_features,
        'n_features': len(selected_features)
    }
    
    selected_path = RESULTS_DIR / "selected_features.json"
    with open(selected_path, 'w') as f:
        json.dump(selected_features_dict, f, indent=2)
    logger.info(f"✓ Saved selected features to {selected_path}")


def main():
    """Main feature selection pipeline"""
    logger.info("="*60)
    logger.info("Starting feature selection pipeline")
    logger.info("="*60)
    
    try:
        # Load processed data
        X, y, feature_names = load_processed_data()
        
        # ReliefF-like ranking
        relieff_df = relieff_approximation(X, y, feature_names)
        
        # Mutual information
        mi_df = compute_mutual_information(X, y, feature_names)
        
        # PCA
        pca_dict = compute_pca(X, feature_names)
        
        # Select features
        selected_features, combined_scores = select_features(relieff_df, mi_df, pca_dict)
        
        # Save results
        save_results(relieff_df, mi_df, pca_dict, selected_features, combined_scores)
        
        # Create verification marker
        marker_path = RESULTS_DIR / ".feature_selection_ok"
        marker_path.touch()
        logger.info(f"✓ Created verification marker {marker_path}")
        
        logger.info("="*60)
        logger.info("✓ Feature selection completed successfully!")
        logger.info("="*60)
        
        return 0
        
    except Exception as e:
        logger.error(f"✗ Error during feature selection: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
