"""
Risk prediction module for hearing loss.

This module provides multi-level risk scoring (variant, gene, sample)
using trained ML models.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import joblib

from model.gene_database import get_gene_database, is_hearing_loss_gene


@dataclass
class VariantScore:
    """Score for individual variant."""
    chrom: str
    pos: int
    ref: str
    alt: str
    gene: str
    pathogenicity_score: float
    population_frequency: float
    clinical_significance: str
    contribution_to_risk: float


@dataclass
class GeneScore:
    """Aggregated score for a gene."""
    gene_symbol: str
    variant_count: int
    max_pathogenicity: float
    compound_heterozygous: bool
    homozygous_count: int
    gene_risk_score: float


@dataclass
class SampleRiskPrediction:
    """Complete risk prediction for a sample."""
    sample_id: str
    overall_risk_score: float
    risk_category: str  # "low", "moderate", "high"
    confidence: float
    variant_scores: List[VariantScore]
    gene_scores: List[GeneScore]
    key_findings: List[str]
    recommendations: List[str]


class HearingLossPredictor:
    """Predict hearing loss risk from genomic variants."""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to trained model file
        """
        self.model = None
        self.gene_db = get_gene_database()
        
        if model_path and Path(model_path).exists():
            self.model = joblib.load(model_path)
    
    def predict_risk(
        self,
        variants: List[Dict],
        clinical_features: Optional[Dict] = None
    ) -> SampleRiskPrediction:
        """
        Predict hearing loss risk from variants.
        
        Args:
            variants: List of variant dictionaries
            clinical_features: Optional clinical information
            
        Returns:
            SampleRiskPrediction object
        """
        # Score individual variants
        variant_scores = [self._score_variant(v) for v in variants]
        
        # Aggregate scores by gene
        gene_scores = self._aggregate_gene_scores(variant_scores, variants)
        
        # Calculate overall risk
        overall_score = self._calculate_overall_risk(variant_scores, gene_scores, clinical_features)
        
        # Determine risk category
        risk_category = self._categorize_risk(overall_score)
        
        # Generate findings and recommendations
        key_findings = self._generate_findings(variant_scores, gene_scores)
        recommendations = self._generate_recommendations(risk_category, gene_scores)
        
        # Calculate confidence
        confidence = self._calculate_confidence(variant_scores, gene_scores)
        
        return SampleRiskPrediction(
            sample_id=clinical_features.get("sample_id", "UNKNOWN") if clinical_features else "UNKNOWN",
            overall_risk_score=overall_score,
            risk_category=risk_category,
            confidence=confidence,
            variant_scores=variant_scores,
            gene_scores=gene_scores,
            key_findings=key_findings,
            recommendations=recommendations
        )
    
    def _score_variant(self, variant: Dict) -> VariantScore:
        """Score individual variant for pathogenicity."""
        gene = variant.get("gene", "")
        
        # Extract annotation data
        clinvar_sig = variant.get("clinvar_sig", "").lower()
        gnomad_af = float(variant.get("gnomad_af", 0.01))
        cadd_score = float(variant.get("cadd_phred", 0))
        consequence = variant.get("consequence", "")
        
        # Calculate pathogenicity score (0-1)
        path_score = 0.0
        
        # ClinVar contribution
        if "pathogenic" in clinvar_sig and "benign" not in clinvar_sig:
            path_score += 0.4
        elif "likely_pathogenic" in clinvar_sig:
            path_score += 0.3
        
        # Population frequency (rare variants more likely pathogenic)
        if gnomad_af < 0.0001:
            path_score += 0.2
        elif gnomad_af < 0.001:
            path_score += 0.1
        
        # CADD score
        if cadd_score > 30:
            path_score += 0.2
        elif cadd_score > 20:
            path_score += 0.1
        
        # Consequence
        high_impact = ["stop_gained", "frameshift", "splice_donor", "splice_acceptor"]
        if any(c in consequence for c in high_impact):
            path_score += 0.2
        
        # Hearing loss gene bonus
        if is_hearing_loss_gene(gene):
            path_score += 0.1
        
        path_score = min(path_score, 1.0)
        
        # Contribution to overall risk
        contribution = path_score * (1.0 - gnomad_af) * 0.5
        
        return VariantScore(
            chrom=variant.get("chrom", ""),
            pos=int(variant.get("pos", 0)),
            ref=variant.get("ref", ""),
            alt=variant.get("alt", ""),
            gene=gene,
            pathogenicity_score=path_score,
            population_frequency=gnomad_af,
            clinical_significance=clinvar_sig,
            contribution_to_risk=contribution
        )
    
    def _aggregate_gene_scores(
        self,
        variant_scores: List[VariantScore],
        raw_variants: List[Dict]
    ) -> List[GeneScore]:
        """Aggregate variant scores by gene."""
        gene_variants = {}
        
        for score, raw in zip(variant_scores, raw_variants):
            gene = score.gene
            if not gene:
                continue
            
            if gene not in gene_variants:
                gene_variants[gene] = []
            
            gene_variants[gene].append((score, raw))
        
        gene_scores = []
        
        for gene, variants in gene_variants.items():
            scores = [s for s, _ in variants]
            raws = [r for _, r in variants]
            
            # Check for compound heterozygosity or homozygosity
            genotypes = [r.get("genotype", "0/0") for r in raws]
            has_homozygous = any(gt in ["1/1", "1|1"] for gt in genotypes)
            has_compound_het = sum(1 for gt in genotypes if gt in ["0/1", "1/0", "0|1", "1|0"]) >= 2
            
            max_path = max(s.pathogenicity_score for s in scores)
            
            # Calculate gene-level risk
            gene_risk = max_path
            if has_homozygous:
                gene_risk *= 1.5
            if has_compound_het:
                gene_risk *= 1.3
            
            gene_risk = min(gene_risk, 1.0)
            
            gene_scores.append(GeneScore(
                gene_symbol=gene,
                variant_count=len(variants),
                max_pathogenicity=max_path,
                compound_heterozygous=has_compound_het,
                homozygous_count=sum(1 for gt in genotypes if gt in ["1/1", "1|1"]),
                gene_risk_score=gene_risk
            ))
        
        # Sort by risk score descending
        gene_scores.sort(key=lambda x: x.gene_risk_score, reverse=True)
        
        return gene_scores
    
    def _calculate_overall_risk(
        self,
        variant_scores: List[VariantScore],
        gene_scores: List[GeneScore],
        clinical_features: Optional[Dict]
    ) -> float:
        """Calculate overall sample risk score."""
        if not gene_scores:
            return 0.0
        
        # Weight by top genes
        top_genes_risk = sum(g.gene_risk_score for g in gene_scores[:5]) / 5.0
        
        # Consider variant burden
        high_impact_count = sum(1 for v in variant_scores if v.pathogenicity_score > 0.7)
        burden_factor = min(high_impact_count / 10.0, 0.3)
        
        # Clinical features adjustment (if available)
        clinical_adjustment = 0.0
        if clinical_features:
            family_history = clinical_features.get("family_history", False)
            consanguinity = clinical_features.get("consanguinity", False)
            
            if family_history:
                clinical_adjustment += 0.1
            if consanguinity:
                clinical_adjustment += 0.1
        
        overall_risk = min(top_genes_risk + burden_factor + clinical_adjustment, 1.0)
        
        return overall_risk
    
    def _categorize_risk(self, risk_score: float) -> str:
        """Categorize risk score into low/moderate/high."""
        if risk_score < 0.3:
            return "low"
        elif risk_score < 0.7:
            return "moderate"
        else:
            return "high"
    
    def _calculate_confidence(
        self,
        variant_scores: List[VariantScore],
        gene_scores: List[GeneScore]
    ) -> float:
        """Calculate prediction confidence."""
        if not variant_scores:
            return 0.0
        
        # Higher confidence if:
        # - Variants have ClinVar annotations
        # - Variants in known hearing loss genes
        # - Clear genetic mechanism (homozygous/compound het)
        
        clinvar_annotated = sum(1 for v in variant_scores if v.clinical_significance)
        hl_gene_variants = sum(1 for v in variant_scores if is_hearing_loss_gene(v.gene))
        clear_mechanism = sum(1 for g in gene_scores if g.homozygous_count > 0 or g.compound_heterozygous)
        
        confidence = 0.5  # Base confidence
        confidence += (clinvar_annotated / len(variant_scores)) * 0.2
        confidence += (hl_gene_variants / len(variant_scores)) * 0.2
        confidence += min(clear_mechanism / 3.0, 0.1)
        
        return min(confidence, 1.0)
    
    def _generate_findings(
        self,
        variant_scores: List[VariantScore],
        gene_scores: List[GeneScore]
    ) -> List[str]:
        """Generate key findings from analysis."""
        findings = []
        
        # Top pathogenic variants
        pathogenic = [v for v in variant_scores if v.pathogenicity_score > 0.6]
        if pathogenic:
            findings.append(f"Identified {len(pathogenic)} likely pathogenic variant(s)")
        
        # Top genes
        high_risk_genes = [g for g in gene_scores if g.gene_risk_score > 0.6]
        if high_risk_genes:
            gene_list = ", ".join(g.gene_symbol for g in high_risk_genes[:3])
            findings.append(f"High-risk variants in hearing loss genes: {gene_list}")
        
        # Compound heterozygosity
        compound_het_genes = [g for g in gene_scores if g.compound_heterozygous]
        if compound_het_genes:
            findings.append(f"Compound heterozygosity detected in {compound_het_genes[0].gene_symbol}")
        
        # Homozygous variants
        homozygous_genes = [g for g in gene_scores if g.homozygous_count > 0]
        if homozygous_genes:
            findings.append(f"Homozygous variant(s) in {homozygous_genes[0].gene_symbol}")
        
        return findings
    
    def _generate_recommendations(
        self,
        risk_category: str,
        gene_scores: List[GeneScore]
    ) -> List[str]:
        """Generate clinical recommendations."""
        recommendations = []
        
        if risk_category == "high":
            recommendations.append("Recommend comprehensive audiological evaluation")
            recommendations.append("Consider genetic counseling for family planning")
            recommendations.append("Early intervention and amplification if hearing loss confirmed")
        elif risk_category == "moderate":
            recommendations.append("Recommend audiological monitoring")
            recommendations.append("Consider genetic counseling")
        else:
            recommendations.append("Standard newborn hearing screening protocol")
        
        # Gene-specific recommendations
        if gene_scores:
            top_gene = gene_scores[0].gene_symbol
            gene_info = self.gene_db.get_gene(top_gene)
            
            if gene_info and gene_info.syndromic:
                recommendations.append(f"Screen for {top_gene}-associated syndromes: {', '.join(gene_info.associated_syndromes)}")
        
        return recommendations


def predict_sample_risk(
    variants: List[Dict],
    sample_id: str,
    model_path: Optional[str] = None
) -> Dict:
    """
    Predict hearing loss risk for a sample.
    
    Args:
        variants: List of variant dictionaries
        sample_id: Sample identifier
        model_path: Optional path to trained model
        
    Returns:
        Dictionary with prediction results
        
    Example:
        >>> variants = [{"gene": "GJB2", "clinvar_sig": "Pathogenic", ...}]
        >>> result = predict_sample_risk(variants, "SAMPLE001")
        >>> print(result["risk_category"])
        high
    """
    predictor = HearingLossPredictor(model_path)
    
    clinical_features = {"sample_id": sample_id}
    prediction = predictor.predict_risk(variants, clinical_features)
    
    return {
        "sample_id": prediction.sample_id,
        "risk_score": prediction.overall_risk_score,
        "risk_category": prediction.risk_category,
        "confidence": prediction.confidence,
        "key_findings": prediction.key_findings,
        "recommendations": prediction.recommendations,
        "top_genes": [
            {
                "gene": g.gene_symbol,
                "risk_score": g.gene_risk_score,
                "variant_count": g.variant_count,
                "compound_heterozygous": g.compound_heterozygous,
                "homozygous_count": g.homozygous_count
            }
            for g in prediction.gene_scores[:5]
        ],
        "pathogenic_variants": [
            {
                "variant": f"{v.chrom}:{v.pos}:{v.ref}>{v.alt}",
                "gene": v.gene,
                "pathogenicity": v.pathogenicity_score,
                "significance": v.clinical_significance
            }
            for v in prediction.variant_scores if v.pathogenicity_score > 0.6
        ]
    }
