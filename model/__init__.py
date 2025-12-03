"""Enhanced AI/ML model for hearing loss prediction."""

from model.gene_database import (
    HearingLossGeneDatabase,
    HearingLossGene,
    get_gene_database,
    get_gene_info,
    is_hearing_loss_gene,
    get_hearing_loss_genes
)

from model.predictor import (
    HearingLossPredictor,
    SampleRiskPrediction,
    VariantScore,
    GeneScore,
    predict_sample_risk
)

from model.explainer import explain_prediction
from model.training import train_hearing_loss_model
from model.evaluation import evaluate_model

__all__ = [
    "HearingLossGeneDatabase",
    "HearingLossGene",
    "get_gene_database",
    "get_gene_info",
    "is_hearing_loss_gene",
    "get_hearing_loss_genes",
    "HearingLossPredictor",
    "SampleRiskPrediction",
    "VariantScore",
    "GeneScore",
    "predict_sample_risk",
    "explain_prediction",
    "train_hearing_loss_model",
    "evaluate_model"
]
