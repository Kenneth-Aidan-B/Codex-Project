"""
Bioinformatics pipeline for genomic sequencing data processing.

This package provides end-to-end processing of FASTQ files through
alignment, variant calling, and annotation for hearing loss screening.
"""

from pipeline.config import (
    get_config,
    PipelineConfig,
    ReferenceGenomeConfig,
    QualityControlConfig,
    AlignmentConfig,
    VariantCallingConfig,
    AnnotationConfig,
    HearingLossGenesConfig,
)

from pipeline.fastq_processor import (
    FastqProcessor,
    ReadQualityMetrics,
    process_fastq_sample,
)

from pipeline.alignment import (
    AlignmentEngine,
    AlignmentMetrics,
    align_sample,
)

from pipeline.variant_caller import (
    VariantCaller,
    VariantCallStats,
    call_variants_for_sample,
)

from pipeline.annotator import (
    VariantAnnotator,
    VariantAnnotation,
    annotate_sample_vcf,
)

from pipeline.vcf_parser import (
    VCFParser,
    VCFVariant,
    extract_hearing_loss_variants,
    write_filtered_vcf,
    compare_vcfs,
)


__version__ = "1.0.0"
__author__ = "Genomic Hearing Screening Platform Team"

__all__ = [
    # Config
    "get_config",
    "PipelineConfig",
    "ReferenceGenomeConfig",
    "QualityControlConfig",
    "AlignmentConfig",
    "VariantCallingConfig",
    "AnnotationConfig",
    "HearingLossGenesConfig",
    
    # FASTQ processing
    "FastqProcessor",
    "ReadQualityMetrics",
    "process_fastq_sample",
    
    # Alignment
    "AlignmentEngine",
    "AlignmentMetrics",
    "align_sample",
    
    # Variant calling
    "VariantCaller",
    "VariantCallStats",
    "call_variants_for_sample",
    
    # Annotation
    "VariantAnnotator",
    "VariantAnnotation",
    "annotate_sample_vcf",
    
    # VCF parsing
    "VCFParser",
    "VCFVariant",
    "extract_hearing_loss_variants",
    "write_filtered_vcf",
    "compare_vcfs",
]


def run_complete_pipeline(
    fastq_r1: str,
    fastq_r2: str,
    sample_id: str,
    output_dir: str
) -> dict:
    """
    Run complete bioinformatics pipeline from FASTQ to annotated VCF.
    
    Args:
        fastq_r1: Path to R1 FASTQ file
        fastq_r2: Path to R2 FASTQ file
        sample_id: Sample identifier
        output_dir: Output directory for all results
        
    Returns:
        Dictionary with all pipeline results
        
    Example:
        >>> results = run_complete_pipeline(
        ...     "sample_R1.fastq.gz",
        ...     "sample_R2.fastq.gz",
        ...     "SAMPLE001",
        ...     "/output"
        ... )
        >>> print(results["final_vcf"])
        /output/variants/SAMPLE001.annotated.vcf
    """
    from pathlib import Path
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {
        "sample_id": sample_id,
        "fastq_r1": fastq_r1,
        "fastq_r2": fastq_r2,
    }
    
    # Step 1: FASTQ QC and filtering
    print(f"[{sample_id}] Step 1/4: FASTQ quality control...")
    qc_dir = str(output_path / "qc")
    qc_results = process_fastq_sample(fastq_r1, fastq_r2, qc_dir)
    results["qc"] = qc_results
    
    # Step 2: Alignment
    print(f"[{sample_id}] Step 2/4: Alignment to reference genome...")
    alignment_dir = str(output_path / "alignment")
    alignment_results = align_sample(
        qc_results["r1_path"],
        qc_results["r2_path"],
        alignment_dir,
        sample_id
    )
    results["alignment"] = alignment_results
    
    # Step 3: Variant calling
    print(f"[{sample_id}] Step 3/4: Variant calling...")
    variants_dir = str(output_path / "variants")
    variant_results = call_variants_for_sample(
        alignment_results["final_bam"],
        variants_dir,
        sample_id
    )
    results["variants"] = variant_results
    
    # Step 4: Annotation
    print(f"[{sample_id}] Step 4/4: Variant annotation...")
    annotated_vcf = str(output_path / "variants" / f"{sample_id}.annotated.vcf")
    annotation_json = str(output_path / "variants" / f"{sample_id}.annotations.json")
    annotation_results = annotate_sample_vcf(
        variant_results["final_vcf"],
        annotated_vcf,
        annotation_json
    )
    results["annotation"] = annotation_results
    results["final_vcf"] = annotated_vcf
    
    print(f"[{sample_id}] Pipeline complete!")
    print(f"  - Total variants: {variant_results['stats']['total_variants']}")
    print(f"  - Hearing loss gene variants: {annotation_results['hearing_loss_variants']}")
    print(f"  - Pathogenic variants: {annotation_results['pathogenic_variants']}")
    
    return results
