"""
Variant annotation module.

This module provides functionality for annotating variants with information
from clinical databases (ClinVar, gnomAD, dbNSFP, OMIM).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import json

from pipeline.config import get_config, AnnotationConfig


@dataclass
class VariantAnnotation:
    """Annotation information for a variant."""
    
    # Variant identifiers
    chrom: str
    pos: int
    ref: str
    alt: str
    
    # Gene information
    gene_symbol: str = ""
    gene_id: str = ""
    transcript_id: str = ""
    
    # Functional prediction
    consequence: str = ""
    impact: str = ""
    biotype: str = ""
    
    # Position information
    cdna_position: str = ""
    cds_position: str = ""
    protein_position: str = ""
    amino_acids: str = ""
    codons: str = ""
    
    # HGVS notation
    hgvs_c: str = ""
    hgvs_p: str = ""
    
    # Pathogenicity predictions
    sift_score: Optional[float] = None
    sift_pred: str = ""
    polyphen_score: Optional[float] = None
    polyphen_pred: str = ""
    cadd_phred: Optional[float] = None
    
    # Population frequencies
    gnomad_af: Optional[float] = None
    gnomad_af_afr: Optional[float] = None
    gnomad_af_amr: Optional[float] = None
    gnomad_af_eas: Optional[float] = None
    gnomad_af_nfe: Optional[float] = None
    gnomad_af_sas: Optional[float] = None
    
    # Clinical significance
    clinvar_sig: str = ""
    clinvar_id: str = ""
    clinvar_disease: str = ""
    
    # OMIM
    omim_id: str = ""
    omim_phenotype: str = ""
    
    # Hearing loss specific
    hearing_loss_gene: bool = False
    hearing_loss_association: str = ""
    
    def to_dict(self) -> Dict:
        """Convert annotation to dictionary."""
        return {
            "variant": f"{self.chrom}:{self.pos}:{self.ref}>{self.alt}",
            "gene": {
                "symbol": self.gene_symbol,
                "id": self.gene_id,
                "transcript": self.transcript_id,
            },
            "consequence": {
                "type": self.consequence,
                "impact": self.impact,
                "hgvs_c": self.hgvs_c,
                "hgvs_p": self.hgvs_p,
            },
            "pathogenicity": {
                "sift": {"score": self.sift_score, "prediction": self.sift_pred},
                "polyphen": {"score": self.polyphen_score, "prediction": self.polyphen_pred},
                "cadd_phred": self.cadd_phred,
            },
            "population": {
                "gnomad_af": self.gnomad_af,
                "gnomad_af_popmax": max(filter(None, [
                    self.gnomad_af_afr, self.gnomad_af_amr,
                    self.gnomad_af_eas, self.gnomad_af_nfe, self.gnomad_af_sas
                ] or [0])),
            },
            "clinical": {
                "clinvar_significance": self.clinvar_sig,
                "clinvar_id": self.clinvar_id,
                "disease": self.clinvar_disease,
            },
            "hearing_loss": {
                "is_hearing_gene": self.hearing_loss_gene,
                "association": self.hearing_loss_association,
            }
        }


class VariantAnnotator:
    """Annotate variants with clinical and population databases."""
    
    def __init__(self, config: Optional[AnnotationConfig] = None):
        """
        Initialize variant annotator.
        
        Args:
            config: Annotation configuration (uses default if None)
        """
        self.config = config or get_config().annotation
        self.hearing_genes_config = get_config().hearing_genes
        self.hearing_loss_genes = set(self.hearing_genes_config.get_all_genes())
    
    def annotate_vcf(
        self,
        input_vcf: str,
        output_vcf: str,
        include_vep: bool = True
    ) -> Dict[str, int]:
        """
        Annotate VCF file with all databases.
        
        Args:
            input_vcf: Input VCF file path
            output_vcf: Output annotated VCF path
            include_vep: Whether to run VEP annotation
            
        Returns:
            Dictionary with annotation statistics
            
        Raises:
            FileNotFoundError: If input VCF not found
        """
        if not Path(input_vcf).exists():
            raise FileNotFoundError(f"VCF file not found: {input_vcf}")
        
        Path(output_vcf).parent.mkdir(parents=True, exist_ok=True)
        
        # Parse input VCF
        variants = self._parse_vcf(input_vcf)
        
        # Annotate each variant
        annotated_variants = []
        for variant in variants:
            annotation = self.annotate_variant(
                variant["chrom"],
                variant["pos"],
                variant["ref"],
                variant["alt"]
            )
            annotated_variants.append(annotation)
        
        # Write annotated VCF
        self._write_annotated_vcf(output_vcf, annotated_variants)
        
        stats = {
            "total_variants": len(annotated_variants),
            "annotated": len(annotated_variants),
            "hearing_loss_genes": sum(1 for v in annotated_variants if v.hearing_loss_gene),
            "clinvar_pathogenic": sum(1 for v in annotated_variants if "pathogenic" in v.clinvar_sig.lower()),
        }
        
        return stats
    
    def annotate_variant(
        self,
        chrom: str,
        pos: int,
        ref: str,
        alt: str
    ) -> VariantAnnotation:
        """
        Annotate a single variant.
        
        Args:
            chrom: Chromosome
            pos: Position
            ref: Reference allele
            alt: Alternate allele
            
        Returns:
            VariantAnnotation object
        """
        annotation = VariantAnnotation(
            chrom=chrom,
            pos=pos,
            ref=ref,
            alt=alt
        )
        
        # Mock annotation for demonstration
        # In production, would query actual databases
        
        # Determine gene (mock logic)
        gene_symbol = self._determine_gene(chrom, pos)
        annotation.gene_symbol = gene_symbol
        annotation.gene_id = f"ENSG_{gene_symbol}"
        annotation.transcript_id = f"ENST_{gene_symbol}"
        
        # Check if hearing loss gene
        annotation.hearing_loss_gene = gene_symbol in self.hearing_loss_genes
        if annotation.hearing_loss_gene:
            annotation.hearing_loss_association = self._get_hearing_loss_info(gene_symbol)
        
        # Consequence prediction
        annotation.consequence = self._predict_consequence(ref, alt)
        annotation.impact = self._determine_impact(annotation.consequence)
        
        # HGVS notation
        annotation.hgvs_c = f"c.{pos}{ref}>{alt}"
        if annotation.consequence in ["missense_variant", "nonsense_variant"]:
            annotation.hgvs_p = f"p.{self._predict_aa_change(pos)}"
        
        # Pathogenicity scores
        annotation.sift_score = 0.05 if annotation.impact == "HIGH" else 0.15
        annotation.sift_pred = "deleterious" if annotation.sift_score < 0.05 else "tolerated"
        annotation.polyphen_score = 0.95 if annotation.impact == "HIGH" else 0.75
        annotation.polyphen_pred = "probably_damaging" if annotation.polyphen_score > 0.85 else "possibly_damaging"
        annotation.cadd_phred = 25.0 if annotation.impact == "HIGH" else 15.0
        
        # Population frequency
        annotation.gnomad_af = 0.0001 if annotation.hearing_loss_gene else 0.01
        annotation.gnomad_af_nfe = 0.0002
        
        # ClinVar
        if annotation.hearing_loss_gene and annotation.impact in ["HIGH", "MODERATE"]:
            annotation.clinvar_sig = "Pathogenic"
            annotation.clinvar_id = f"VCV{pos}"
            annotation.clinvar_disease = "Hearing loss"
        
        return annotation
    
    def _parse_vcf(self, vcf_path: str) -> List[Dict]:
        """Parse VCF file and extract variant information."""
        variants = []
        
        with open(vcf_path, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                
                fields = line.strip().split('\t')
                if len(fields) < 5:
                    continue
                
                variants.append({
                    "chrom": fields[0],
                    "pos": int(fields[1]),
                    "ref": fields[3],
                    "alt": fields[4].split(',')[0],  # Take first alt allele
                })
        
        return variants
    
    def _determine_gene(self, chrom: str, pos: int) -> str:
        """Determine gene symbol for variant (mock logic)."""
        # Mock gene assignment based on known hearing loss gene loci
        gene_map = {
            "chr13": "GJB2",      # 13q12.11
            "chr7": "SLC26A4",    # 7q31.1
            "chr2": "OTOF",       # 2p23.3
            "chr11": "MYO7A",     # 11q13.5
            "chr10": "CDH23",     # 10q22.1
            "chr9": "TMC1",       # 9q21.13
            "chr11": "TECTA",     # 11q23.3
            "chr14": "COCH",      # 14q12
        }
        
        return gene_map.get(chrom, "UNKNOWN")
    
    def _get_hearing_loss_info(self, gene: str) -> str:
        """Get hearing loss association information for gene."""
        associations = {
            "GJB2": "DFNB1 - Connexin 26, most common cause of autosomal recessive hearing loss",
            "SLC26A4": "DFNB4 - Pendred syndrome, enlarged vestibular aqueduct",
            "OTOF": "DFNB9 - Otoferlin, auditory neuropathy spectrum disorder",
            "MYO7A": "USH1B - Usher syndrome type 1B, retinitis pigmentosa",
            "CDH23": "USH1D - Usher syndrome type 1D",
            "TMC1": "DFNB7/11 - Progressive hearing loss",
            "TECTA": "DFNA8/12 - Mid-frequency hearing loss",
            "COCH": "DFNA9 - Progressive hearing loss with vestibular dysfunction",
        }
        
        return associations.get(gene, "Hearing loss associated")
    
    def _predict_consequence(self, ref: str, alt: str) -> str:
        """Predict variant consequence type."""
        if len(ref) == 1 and len(alt) == 1:
            # SNV
            if ref != alt:
                return "missense_variant"
            return "synonymous_variant"
        elif len(ref) > len(alt):
            return "frameshift_variant" if (len(ref) - len(alt)) % 3 != 0 else "inframe_deletion"
        else:
            return "frameshift_variant" if (len(alt) - len(ref)) % 3 != 0 else "inframe_insertion"
    
    def _determine_impact(self, consequence: str) -> str:
        """Determine variant impact level."""
        high_impact = ["stop_gained", "frameshift_variant", "splice_donor_variant", "splice_acceptor_variant"]
        moderate_impact = ["missense_variant", "inframe_deletion", "inframe_insertion"]
        
        if consequence in high_impact:
            return "HIGH"
        elif consequence in moderate_impact:
            return "MODERATE"
        else:
            return "LOW"
    
    def _predict_aa_change(self, pos: int) -> str:
        """Predict amino acid change (mock)."""
        # Mock amino acid change
        aa = ["Ala", "Arg", "Asn", "Asp", "Cys", "Gln", "Glu", "Gly", "His", "Ile"]
        aa_pos = (pos % 500) + 1
        from_aa = aa[pos % len(aa)]
        to_aa = aa[(pos + 1) % len(aa)]
        return f"{from_aa}{aa_pos}{to_aa}"
    
    def _write_annotated_vcf(
        self,
        output_path: str,
        annotations: List[VariantAnnotation]
    ) -> None:
        """Write annotated variants to VCF file."""
        # Write VCF with annotations
        with open(output_path, 'w') as f:
            # Write header
            f.write("##fileformat=VCFv4.2\n")
            f.write("##reference=hg38\n")
            f.write("##INFO=<ID=GENE,Number=1,Type=String,Description=\"Gene symbol\">\n")
            f.write("##INFO=<ID=CONS,Number=1,Type=String,Description=\"Consequence\">\n")
            f.write("##INFO=<ID=IMPACT,Number=1,Type=String,Description=\"Impact\">\n")
            f.write("##INFO=<ID=SIFT,Number=1,Type=Float,Description=\"SIFT score\">\n")
            f.write("##INFO=<ID=POLYPHEN,Number=1,Type=Float,Description=\"PolyPhen score\">\n")
            f.write("##INFO=<ID=CADD,Number=1,Type=Float,Description=\"CADD PHRED score\">\n")
            f.write("##INFO=<ID=GNOMAD_AF,Number=1,Type=Float,Description=\"gnomAD allele frequency\">\n")
            f.write("##INFO=<ID=CLNSIG,Number=1,Type=String,Description=\"ClinVar significance\">\n")
            f.write("##INFO=<ID=HL_GENE,Number=0,Type=Flag,Description=\"Hearing loss gene\">\n")
            f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
            
            # Write variants
            for ann in annotations:
                info_fields = [
                    f"GENE={ann.gene_symbol}",
                    f"CONS={ann.consequence}",
                    f"IMPACT={ann.impact}",
                    f"SIFT={ann.sift_score}",
                    f"POLYPHEN={ann.polyphen_score}",
                    f"CADD={ann.cadd_phred}",
                    f"GNOMAD_AF={ann.gnomad_af}",
                ]
                
                if ann.clinvar_sig:
                    info_fields.append(f"CLNSIG={ann.clinvar_sig}")
                
                if ann.hearing_loss_gene:
                    info_fields.append("HL_GENE")
                
                info_str = ";".join(info_fields)
                
                f.write(f"{ann.chrom}\t{ann.pos}\t.\t{ann.ref}\t{ann.alt}\t.\tPASS\t{info_str}\n")
    
    def get_clinvar_annotation(
        self,
        chrom: str,
        pos: int,
        ref: str,
        alt: str
    ) -> Dict:
        """
        Query ClinVar for variant annotation.
        
        Args:
            chrom: Chromosome
            pos: Position
            ref: Reference allele
            alt: Alternate allele
            
        Returns:
            Dictionary with ClinVar annotation
        """
        # Mock ClinVar query
        return {
            "significance": "Pathogenic",
            "review_status": "criteria provided, multiple submitters, no conflicts",
            "disease": "Nonsyndromic hearing loss",
            "accession": f"VCV{pos}",
        }
    
    def get_gnomad_frequency(
        self,
        chrom: str,
        pos: int,
        ref: str,
        alt: str
    ) -> Dict:
        """
        Query gnomAD for population frequency.
        
        Args:
            chrom: Chromosome
            pos: Position
            ref: Reference allele
            alt: Alternate allele
            
        Returns:
            Dictionary with gnomAD frequencies
        """
        # Mock gnomAD query
        return {
            "af": 0.0001,
            "af_afr": 0.0002,
            "af_amr": 0.00015,
            "af_eas": 0.00008,
            "af_nfe": 0.00012,
            "af_sas": 0.00011,
            "an": 152000,
            "ac": 15,
        }


def annotate_sample_vcf(
    input_vcf: str,
    output_vcf: str,
    output_json: Optional[str] = None
) -> Dict[str, any]:
    """
    Annotate VCF file with all databases and generate report.
    
    Args:
        input_vcf: Input VCF file
        output_vcf: Output annotated VCF
        output_json: Optional path for JSON annotation report
        
    Returns:
        Dictionary with annotation results
        
    Example:
        >>> results = annotate_sample_vcf(
        ...     "sample.vcf",
        ...     "sample.annotated.vcf",
        ...     "annotations.json"
        ... )
        >>> print(results["hearing_loss_variants"])
        5
    """
    annotator = VariantAnnotator()
    
    # Annotate VCF
    stats = annotator.annotate_vcf(input_vcf, output_vcf)
    
    results = {
        "input_vcf": input_vcf,
        "output_vcf": output_vcf,
        "stats": stats,
        "hearing_loss_variants": stats.get("hearing_loss_genes", 0),
        "pathogenic_variants": stats.get("clinvar_pathogenic", 0),
    }
    
    # Generate JSON report if requested
    if output_json:
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=2)
        results["json_report"] = output_json
    
    return results
