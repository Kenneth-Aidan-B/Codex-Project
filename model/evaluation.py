"""Model evaluation metrics."""

from typing import Dict, List, Optional
import numpy as np


def evaluate_model(
    y_true: List,
    y_pred: List,
    y_prob: Optional[List] = None
) -> Dict:
    """
    Evaluate model performance.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels  
        y_prob: Prediction probabilities
        
    Returns:
        Dictionary with evaluation metrics
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    
    if y_prob is not None:
        from sklearn.metrics import roc_auc_score
        try:
            metrics["auc_roc"] = roc_auc_score(y_true, y_prob)
        except:
            metrics["auc_roc"] = 0.0
    
    return metrics
