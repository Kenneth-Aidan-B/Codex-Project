#!/usr/bin/env python3
"""
Model training script for hearing deficiency ML project.
Trains multiple models: RandomForest, SVM, XGBoost, ANN, and a toy Transformer.
Uses 10-fold cross-validation and saves all models and metrics.
"""
import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, roc_auc_score, classification_report)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available, using GradientBoostingClassifier instead")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Random seed
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


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


class SimpleANN(nn.Module):
    """Simple feedforward neural network for binary classification"""
    def __init__(self, input_dim, hidden_dims=[64, 32, 16]):
        super(SimpleANN, self).__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class ANNClassifier:
    """Scikit-learn compatible ANN classifier wrapper"""
    def __init__(self, input_dim, epochs=50, batch_size=32, lr=0.001):
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def fit(self, X, y):
        self.model = SimpleANN(self.input_dim).to(self.device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).unsqueeze(1).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"  Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(loader):.4f}")
        
        return self
    
    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor).cpu().numpy()
        return np.hstack([1 - outputs, outputs])
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)


class ToyTransformerClassifier:
    """
    Toy transformer-like classifier for genomic sequences.
    Uses simple attention mechanism on tokenized features.
    """
    def __init__(self, input_dim, embed_dim=32, num_heads=4, num_layers=2, epochs=30):
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.epochs = epochs
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _build_model(self):
        class TransformerModel(nn.Module):
            def __init__(self, input_dim, embed_dim, num_heads, num_layers):
                super(TransformerModel, self).__init__()
                self.embedding = nn.Linear(input_dim, embed_dim)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=embed_dim, nhead=num_heads, dim_feedforward=128, dropout=0.1
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                self.fc = nn.Linear(embed_dim, 1)
                self.sigmoid = nn.Sigmoid()
            
            def forward(self, x):
                # x shape: (batch, features)
                x = self.embedding(x)  # (batch, embed_dim)
                x = x.unsqueeze(0)  # (1, batch, embed_dim) for transformer
                x = self.transformer(x)
                x = x.squeeze(0)  # (batch, embed_dim)
                x = self.fc(x)
                return self.sigmoid(x)
        
        return TransformerModel(self.input_dim, self.embed_dim, self.num_heads, self.num_layers)
    
    def fit(self, X, y):
        self.model = self._build_model().to(self.device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).unsqueeze(1).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"  Transformer Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(loader):.4f}")
        
        return self
    
    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor).cpu().numpy()
        return np.hstack([1 - outputs, outputs])
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)


def train_random_forest(X, y):
    """Train Random Forest classifier"""
    logger.info("Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=SEED,
        n_jobs=-1
    )
    model.fit(X, y)
    logger.info("✓ Random Forest trained")
    return model


def train_svm(X, y):
    """Train SVM classifier"""
    logger.info("Training SVM...")
    model = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        probability=True,
        random_state=SEED
    )
    model.fit(X, y)
    logger.info("✓ SVM trained")
    return model


def train_xgboost(X, y):
    """Train XGBoost or GradientBoosting classifier"""
    if XGBOOST_AVAILABLE:
        logger.info("Training XGBoost...")
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
            eval_metric='logloss'
        )
    else:
        logger.info("Training GradientBoosting (XGBoost not available)...")
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=SEED
        )
    
    model.fit(X, y)
    logger.info(f"✓ {'XGBoost' if XGBOOST_AVAILABLE else 'GradientBoosting'} trained")
    return model


def train_ann(X, y, input_dim):
    """Train artificial neural network"""
    logger.info("Training ANN...")
    model = ANNClassifier(input_dim=input_dim, epochs=50)
    model.fit(X, y)
    logger.info("✓ ANN trained")
    return model


def train_transformer(X, y, input_dim):
    """Train toy transformer classifier"""
    logger.info("Training Toy Transformer...")
    model = ToyTransformerClassifier(input_dim=input_dim, epochs=30)
    model.fit(X, y)
    logger.info("✓ Toy Transformer trained")
    return model


def evaluate_model(model, X, y, model_name):
    """Evaluate model performance"""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    
    metrics = {
        'model': model_name,
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1': f1_score(y, y_pred, zero_division=0),
        'auc': roc_auc_score(y, y_proba)
    }
    
    logger.info(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['auc']:.4f}")
    
    return metrics


def cross_validate_model(model, X, y, model_name, cv=10):
    """Perform cross-validation"""
    logger.info(f"Performing {cv}-fold cross-validation for {model_name}...")
    
    # For neural network models, we can't use cross_val_score directly
    if isinstance(model, (ANNClassifier, ToyTransformerClassifier)):
        logger.info(f"Skipping CV for {model_name} (neural network)")
        return {'cv_mean_auc': 0.0, 'cv_std_auc': 0.0}
    
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    
    logger.info(f"{model_name} CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    return {
        'cv_mean_auc': cv_scores.mean(),
        'cv_std_auc': cv_scores.std()
    }


def save_model(model, model_name, timestamp):
    """Save trained model"""
    model_dir = MODELS_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # For neural network models, save using torch
    if isinstance(model, (ANNClassifier, ToyTransformerClassifier)):
        model_path = model_dir / f"model_{timestamp}.pt"
        torch.save({
            'model_state_dict': model.model.state_dict(),
            'input_dim': model.input_dim,
            'config': {
                'epochs': model.epochs if hasattr(model, 'epochs') else 50,
                'type': model_name
            }
        }, model_path)
        logger.info(f"✓ Saved {model_name} to {model_path}")
        
        # Also save as latest
        latest_path = model_dir / "model_latest.pt"
        torch.save({
            'model_state_dict': model.model.state_dict(),
            'input_dim': model.input_dim,
            'config': {
                'epochs': model.epochs if hasattr(model, 'epochs') else 50,
                'type': model_name
            }
        }, latest_path)
    else:
        model_path = model_dir / f"model_{timestamp}.joblib"
        joblib.dump(model, model_path)
        logger.info(f"✓ Saved {model_name} to {model_path}")
        
        # Also save as latest
        latest_path = model_dir / "model_latest.joblib"
        joblib.dump(model, latest_path)


def main():
    """Main training pipeline"""
    logger.info("="*60)
    logger.info("Starting model training pipeline")
    logger.info("="*60)
    
    try:
        # Load data
        X, y, feature_names = load_processed_data()
        input_dim = X.shape[1]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Dictionary to store all models
        models = {}
        all_metrics = []
        
        # Train Random Forest
        rf_model = train_random_forest(X, y)
        models['RandomForest'] = rf_model
        rf_metrics = evaluate_model(rf_model, X, y, 'RandomForest')
        rf_cv = cross_validate_model(rf_model, X, y, 'RandomForest')
        rf_metrics.update(rf_cv)
        all_metrics.append(rf_metrics)
        save_model(rf_model, 'RandomForest', timestamp)
        
        # Train SVM
        svm_model = train_svm(X, y)
        models['SVM'] = svm_model
        svm_metrics = evaluate_model(svm_model, X, y, 'SVM')
        svm_cv = cross_validate_model(svm_model, X, y, 'SVM')
        svm_metrics.update(svm_cv)
        all_metrics.append(svm_metrics)
        save_model(svm_model, 'SVM', timestamp)
        
        # Train XGBoost/GradientBoosting
        xgb_model = train_xgboost(X, y)
        model_name = 'XGBoost' if XGBOOST_AVAILABLE else 'GradientBoosting'
        models[model_name] = xgb_model
        xgb_metrics = evaluate_model(xgb_model, X, y, model_name)
        xgb_cv = cross_validate_model(xgb_model, X, y, model_name)
        xgb_metrics.update(xgb_cv)
        all_metrics.append(xgb_metrics)
        save_model(xgb_model, model_name, timestamp)
        
        # Train ANN
        ann_model = train_ann(X, y, input_dim)
        models['ANN'] = ann_model
        ann_metrics = evaluate_model(ann_model, X, y, 'ANN')
        ann_metrics.update({'cv_mean_auc': 0.0, 'cv_std_auc': 0.0})
        all_metrics.append(ann_metrics)
        save_model(ann_model, 'ANN', timestamp)
        
        # Train Transformer
        transformer_model = train_transformer(X, y, input_dim)
        models['Transformer'] = transformer_model
        transformer_metrics = evaluate_model(transformer_model, X, y, 'Transformer')
        transformer_metrics.update({'cv_mean_auc': 0.0, 'cv_std_auc': 0.0})
        all_metrics.append(transformer_metrics)
        save_model(transformer_model, 'Transformer', timestamp)
        
        # Save metrics
        metrics_df = pd.DataFrame(all_metrics)
        metrics_path = RESULTS_DIR / "metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        logger.info(f"✓ Saved metrics to {metrics_path}")
        
        # Create verification marker
        marker_path = RESULTS_DIR / ".train_ok"
        marker_path.touch()
        logger.info(f"✓ Created verification marker {marker_path}")
        
        logger.info("="*60)
        logger.info("✓ Model training completed successfully!")
        logger.info("="*60)
        
        return 0
        
    except Exception as e:
        logger.error(f"✗ Error during training: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
