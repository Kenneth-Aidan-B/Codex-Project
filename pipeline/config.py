"""
Configuration module for bioinformatics pipeline.

This module provides configuration settings for sequencing data processing,
including reference genomes, quality thresholds, and tool parameters.
"""

import os
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass, field


@dataclass
class ReferenceGenomeConfig:
    """Configuration for reference genome."""
    
    name: str = "hg38"
    fasta_path: str = "/data/references/hg38.fa"
    bwa_index: str = "/data/references/hg38.fa"
    known_sites: str = "/data/references/dbsnp.vcf.gz"


@dataclass
class QualityControlConfig:
    """Quality control thresholds."""
    
    min_read_length: int = 50
    min_quality_score: float = 20.0
    max_n_content: float = 0.1  # 10% N bases
    min_gc_content: float = 0.30
    max_gc_content: float = 0.70


@dataclass
class AlignmentConfig:
    """Alignment parameters for BWA-MEM."""
    
    threads: int = 8
    min_seed_length: int = 19
    band_width: int = 100
    off_diagonal_x_dropoff: int = 100
    internal_seeds_length: int = 1.5
    skip_seed_threshold: int = 500
    drop_chain_threshold: float = 0.5
    max_mate_rescue_rounds: int = 50
    skip_pairing: bool = False
    mark_shorter_splits: bool = True


@dataclass
class VariantCallingConfig:
    """Variant calling parameters."""
    
    min_mapping_quality: int = 20
    min_base_quality: int = 20
    min_coverage: int = 10
    max_coverage: int = 1000
    min_alternate_fraction: float = 0.2
    ploidy: int = 2
    skip_indels: bool = False
    call_confidence_threshold: float = 30.0


@dataclass
class AnnotationConfig:
    """Variant annotation settings."""
    
    # ClinVar
    clinvar_vcf: str = "/data/annotations/clinvar.vcf.gz"
    clinvar_api_url: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    
    # gnomAD
    gnomad_vcf: str = "/data/annotations/gnomad.genomes.vcf.gz"
    gnomad_api_url: str = "https://gnomad.broadinstitute.org/api"
    
    # dbNSFP
    dbnsfp_db: str = "/data/annotations/dbNSFP.txt.gz"
    
    # OMIM
    omim_api_key: str = os.getenv("OMIM_API_KEY", "")
    omim_api_url: str = "https://api.omim.org/api/"
    
    # VEP (Variant Effect Predictor)
    vep_cache: str = "/data/annotations/vep_cache"
    vep_species: str = "homo_sapiens"
    vep_assembly: str = "GRCh38"
    
    # Annotation fields to extract
    extract_fields: list = field(default_factory=lambda: [
        "SYMBOL", "Gene", "Feature", "Consequence",
        "cDNA_position", "CDS_position", "Protein_position",
        "Amino_acids", "Codons", "IMPACT", "STRAND",
        "SIFT", "PolyPhen", "CADD_PHRED", "gnomAD_AF",
        "ClinVar_CLNSIG", "HGVS_c", "HGVS_p"
    ])


@dataclass
class HearingLossGenesConfig:
    """Configuration for hearing loss gene panel."""
    
    # Primary hearing loss genes
    primary_genes: list = field(default_factory=lambda: [
        "GJB2",      # Connexin 26 (most common)
        "SLC26A4",   # Pendred syndrome
        "OTOF",      # Auditory neuropathy
        "MYO7A",     # Usher syndrome type 1B
        "CDH23",     # Usher syndrome type 1D
        "TMC1",      # Progressive hearing loss
        "TECTA",     # Mid-frequency hearing loss
        "COCH",      # DFNA9
        "MYO15A",    # DFNB3
        "PCDH15",    # Usher syndrome type 1F
    ])
    
    # Extended panel (100+ genes)
    extended_genes: list = field(default_factory=lambda: [
        # Add more genes as needed
        "STRC", "ESPN", "MYO6", "ACTG1", "POU3F4", "COL11A2",
        "TMPRSS3", "LOXHD1", "OTOA", "OTOG", "OTOGL", "TJP2",
        "TRIOBP", "USH1C", "USH1G", "USH2A", "DFNB59", "PJVK",
        "MARVELD2", "LHFPL5", "GIPC3", "RDX", "CIB2", "ILDR1",
        "ESRRB", "GPSM2", "WHRN", "CLRN1", "HARS1", "LARS1",
    ])
    
    # OMIM IDs for hearing loss
    omim_hearing_loss: list = field(default_factory=lambda: [
        "220290",  # Usher syndrome, type 1
        "276903",  # Usher syndrome, type 2
        "274600",  # Usher syndrome, type 3
        "600316",  # Pendred syndrome
        "220700",  # Jervell and Lange-Nielsen syndrome
    ])
    
    def get_all_genes(self) -> list:
        """Get complete list of all hearing loss genes."""
        return list(set(self.primary_genes + self.extended_genes))


@dataclass
class PipelineConfig:
    """Main pipeline configuration."""
    
    # Sub-configurations
    reference: ReferenceGenomeConfig = field(default_factory=ReferenceGenomeConfig)
    qc: QualityControlConfig = field(default_factory=QualityControlConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    variant_calling: VariantCallingConfig = field(default_factory=VariantCallingConfig)
    annotation: AnnotationConfig = field(default_factory=AnnotationConfig)
    hearing_genes: HearingLossGenesConfig = field(default_factory=HearingLossGenesConfig)
    
    # General settings
    output_dir: str = "./data/pipeline_output"
    temp_dir: str = "/tmp/pipeline"
    log_level: str = "INFO"
    keep_intermediate_files: bool = False
    
    # Performance
    max_threads: int = 16
    max_memory_gb: int = 32
    
    # Security
    enable_encryption: bool = True
    encryption_key_path: str = os.getenv("ENCRYPTION_KEY_PATH", "")
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.temp_dir).mkdir(parents=True, exist_ok=True)


# Global configuration instance
_config: PipelineConfig = None


def get_config() -> PipelineConfig:
    """
    Get the global pipeline configuration.
    
    Returns:
        PipelineConfig instance
    """
    global _config
    if _config is None:
        _config = PipelineConfig()
    return _config


def load_config_from_file(config_path: str) -> PipelineConfig:
    """
    Load configuration from JSON or YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        PipelineConfig instance
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config format is invalid
    """
    import json
    
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        if config_path.endswith('.json'):
            config_dict = json.load(f)
        elif config_path.endswith(('.yaml', '.yml')):
            try:
                import yaml
                config_dict = yaml.safe_load(f)
            except ImportError:
                raise ImportError("PyYAML required for YAML config files")
        else:
            raise ValueError(f"Unsupported config format: {config_path}")
    
    return PipelineConfig(**config_dict)


def update_config(updates: Dict[str, Any]) -> None:
    """
    Update global configuration with new values.
    
    Args:
        updates: Dictionary of configuration updates
        
    Example:
        >>> update_config({"alignment": {"threads": 16}})
    """
    global _config
    config = get_config()
    
    for key, value in updates.items():
        if hasattr(config, key):
            if isinstance(value, dict) and hasattr(getattr(config, key), '__dict__'):
                # Update nested config
                nested_config = getattr(config, key)
                for nested_key, nested_value in value.items():
                    if hasattr(nested_config, nested_key):
                        setattr(nested_config, nested_key, nested_value)
            else:
                setattr(config, key, value)


# Tool paths (can be overridden by environment variables)
TOOLS = {
    "bwa": os.getenv("BWA_PATH", "bwa"),
    "samtools": os.getenv("SAMTOOLS_PATH", "samtools"),
    "bcftools": os.getenv("BCFTOOLS_PATH", "bcftools"),
    "fastqc": os.getenv("FASTQC_PATH", "fastqc"),
    "gatk": os.getenv("GATK_PATH", "gatk"),
    "vep": os.getenv("VEP_PATH", "vep"),
    "tabix": os.getenv("TABIX_PATH", "tabix"),
    "bgzip": os.getenv("BGZIP_PATH", "bgzip"),
}


def get_tool_path(tool_name: str) -> str:
    """
    Get the path to a bioinformatics tool.
    
    Args:
        tool_name: Name of the tool (e.g., 'bwa', 'samtools')
        
    Returns:
        Path to the tool executable
        
    Raises:
        ValueError: If tool is not configured
    """
    if tool_name not in TOOLS:
        raise ValueError(f"Unknown tool: {tool_name}")
    return TOOLS[tool_name]
