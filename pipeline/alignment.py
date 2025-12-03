"""
Sequence alignment module using BWA-MEM.

This module provides functionality for aligning sequencing reads
to a reference genome using BWA-MEM algorithm.
"""

import subprocess
import os
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import json

from pipeline.config import get_config, AlignmentConfig, get_tool_path


@dataclass
class AlignmentMetrics:
    """Metrics from sequence alignment."""
    
    total_reads: int = 0
    mapped_reads: int = 0
    properly_paired: int = 0
    singletons: int = 0
    duplicates: int = 0
    mean_coverage: float = 0.0
    mean_insert_size: float = 0.0
    mapping_quality_avg: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert metrics to dictionary."""
        return {
            "total_reads": self.total_reads,
            "mapped_reads": self.mapped_reads,
            "mapping_rate": round(self.mapped_reads / max(self.total_reads, 1), 4),
            "properly_paired": self.properly_paired,
            "singletons": self.singletons,
            "duplicates": self.duplicates,
            "mean_coverage": round(self.mean_coverage, 2),
            "mean_insert_size": round(self.mean_insert_size, 2),
            "mapping_quality_avg": round(self.mapping_quality_avg, 2),
        }


class AlignmentEngine:
    """Perform sequence alignment using BWA-MEM."""
    
    def __init__(self, config: Optional[AlignmentConfig] = None):
        """
        Initialize alignment engine.
        
        Args:
            config: Alignment configuration (uses default if None)
        """
        self.config = config or get_config().alignment
        self.reference_config = get_config().reference
    
    def align_paired_end(
        self,
        fastq_r1: str,
        fastq_r2: str,
        output_bam: str,
        sample_id: str = "SAMPLE",
        read_group: Optional[Dict[str, str]] = None
    ) -> AlignmentMetrics:
        """
        Align paired-end reads to reference genome.
        
        Args:
            fastq_r1: Path to R1 FASTQ file
            fastq_r2: Path to R2 FASTQ file
            output_bam: Output BAM file path
            sample_id: Sample identifier
            read_group: Optional read group information
            
        Returns:
            AlignmentMetrics object
            
        Raises:
            FileNotFoundError: If input files or reference not found
            RuntimeError: If alignment fails
        """
        # Validate inputs
        for path in [fastq_r1, fastq_r2]:
            if not Path(path).exists():
                raise FileNotFoundError(f"FASTQ file not found: {path}")
        
        if not Path(self.reference_config.bwa_index).exists():
            raise FileNotFoundError(f"BWA index not found: {self.reference_config.bwa_index}")
        
        # Create output directory
        Path(output_bam).parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare read group string
        rg_str = self._prepare_read_group(sample_id, read_group)
        
        # Build BWA-MEM command
        bwa_cmd = self._build_bwa_command(fastq_r1, fastq_r2, rg_str)
        
        # Execute alignment (mock for this implementation)
        metrics = self._execute_alignment_mock(
            bwa_cmd, output_bam, fastq_r1, fastq_r2
        )
        
        return metrics
    
    def align_single_end(
        self,
        fastq: str,
        output_bam: str,
        sample_id: str = "SAMPLE",
        read_group: Optional[Dict[str, str]] = None
    ) -> AlignmentMetrics:
        """
        Align single-end reads to reference genome.
        
        Args:
            fastq: Path to FASTQ file
            output_bam: Output BAM file path
            sample_id: Sample identifier
            read_group: Optional read group information
            
        Returns:
            AlignmentMetrics object
        """
        if not Path(fastq).exists():
            raise FileNotFoundError(f"FASTQ file not found: {fastq}")
        
        Path(output_bam).parent.mkdir(parents=True, exist_ok=True)
        
        rg_str = self._prepare_read_group(sample_id, read_group)
        bwa_cmd = self._build_bwa_command(fastq, None, rg_str)
        
        metrics = self._execute_alignment_mock(bwa_cmd, output_bam, fastq)
        
        return metrics
    
    def _prepare_read_group(
        self,
        sample_id: str,
        read_group: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Prepare read group string for BAM header.
        
        Args:
            sample_id: Sample identifier
            read_group: Optional read group information
            
        Returns:
            Read group string for BWA -R parameter
        """
        rg_dict = {
            "ID": read_group.get("id", "1") if read_group else "1",
            "SM": sample_id,
            "PL": read_group.get("platform", "ILLUMINA") if read_group else "ILLUMINA",
            "LB": read_group.get("library", sample_id) if read_group else sample_id,
            "PU": read_group.get("platform_unit", "unit1") if read_group else "unit1",
        }
        
        rg_str = "@RG\\t" + "\\t".join([f"{k}:{v}" for k, v in rg_dict.items()])
        return rg_str
    
    def _build_bwa_command(
        self,
        fastq_r1: str,
        fastq_r2: Optional[str],
        read_group: str
    ) -> list:
        """
        Build BWA-MEM command.
        
        Args:
            fastq_r1: R1 FASTQ path
            fastq_r2: R2 FASTQ path (None for single-end)
            read_group: Read group string
            
        Returns:
            List of command components
        """
        cmd = [
            get_tool_path("bwa"),
            "mem",
            "-t", str(self.config.threads),
            "-R", read_group,
            "-M",  # Mark shorter split hits as secondary
            "-k", str(self.config.min_seed_length),
            "-w", str(self.config.band_width),
            self.reference_config.bwa_index,
            fastq_r1,
        ]
        
        if fastq_r2:
            cmd.append(fastq_r2)
        
        return cmd
    
    def _execute_alignment_mock(
        self,
        bwa_cmd: list,
        output_bam: str,
        fastq_r1: str,
        fastq_r2: Optional[str] = None
    ) -> AlignmentMetrics:
        """
        Execute alignment (mock implementation).
        
        In production, this would:
        1. Run BWA-MEM
        2. Pipe to samtools view for BAM conversion
        3. Sort and index BAM
        4. Calculate alignment metrics
        
        For this implementation, we create mock BAM and metrics.
        
        Args:
            bwa_cmd: BWA command list
            output_bam: Output BAM path
            fastq_r1: R1 FASTQ path
            fastq_r2: R2 FASTQ path
            
        Returns:
            AlignmentMetrics with mock data
        """
        # Create mock BAM file
        Path(output_bam).touch()
        
        # Create mock BAM index
        Path(f"{output_bam}.bai").touch()
        
        # Generate mock metrics
        metrics = AlignmentMetrics(
            total_reads=1000000,
            mapped_reads=950000,
            properly_paired=900000 if fastq_r2 else 0,
            singletons=50000 if fastq_r2 else 0,
            duplicates=50000,
            mean_coverage=30.5,
            mean_insert_size=350.0 if fastq_r2 else 0.0,
            mapping_quality_avg=42.5,
        )
        
        return metrics
    
    def sort_and_index_bam(self, input_bam: str, output_bam: str) -> None:
        """
        Sort and index BAM file.
        
        Args:
            input_bam: Input BAM path
            output_bam: Output sorted BAM path
            
        Raises:
            FileNotFoundError: If input BAM not found
            RuntimeError: If sorting fails
        """
        if not Path(input_bam).exists():
            raise FileNotFoundError(f"BAM file not found: {input_bam}")
        
        # Mock implementation - create sorted BAM and index
        Path(output_bam).touch()
        Path(f"{output_bam}.bai").touch()
    
    def mark_duplicates(
        self,
        input_bam: str,
        output_bam: str,
        metrics_file: str
    ) -> Dict[str, int]:
        """
        Mark duplicate reads in BAM file.
        
        Args:
            input_bam: Input BAM path
            output_bam: Output BAM with marked duplicates
            metrics_file: Path for duplicate metrics output
            
        Returns:
            Dictionary with duplicate statistics
            
        Raises:
            FileNotFoundError: If input BAM not found
        """
        if not Path(input_bam).exists():
            raise FileNotFoundError(f"BAM file not found: {input_bam}")
        
        Path(output_bam).parent.mkdir(parents=True, exist_ok=True)
        
        # Mock implementation
        Path(output_bam).touch()
        Path(f"{output_bam}.bai").touch()
        
        duplicate_stats = {
            "total_reads": 1000000,
            "duplicate_reads": 50000,
            "duplicate_rate": 0.05,
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(duplicate_stats, f, indent=2)
        
        return duplicate_stats
    
    def calculate_coverage(
        self,
        bam_file: str,
        regions: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Calculate coverage statistics for BAM file.
        
        Args:
            bam_file: BAM file path
            regions: Optional BED file with regions of interest
            
        Returns:
            Dictionary with coverage statistics
            
        Raises:
            FileNotFoundError: If BAM file not found
        """
        if not Path(bam_file).exists():
            raise FileNotFoundError(f"BAM file not found: {bam_file}")
        
        # Mock coverage statistics
        coverage_stats = {
            "mean_coverage": 30.5,
            "median_coverage": 32.0,
            "pct_bases_10x": 0.95,
            "pct_bases_20x": 0.90,
            "pct_bases_30x": 0.75,
        }
        
        return coverage_stats
    
    def validate_bam(self, bam_file: str) -> Tuple[bool, str]:
        """
        Validate BAM file integrity.
        
        Args:
            bam_file: BAM file path
            
        Returns:
            Tuple of (is_valid, message)
        """
        if not Path(bam_file).exists():
            return False, f"BAM file not found: {bam_file}"
        
        # Check for index
        if not Path(f"{bam_file}.bai").exists():
            return False, "BAM index (.bai) not found"
        
        # In production, would use samtools quickcheck
        return True, "BAM file is valid"


def align_sample(
    fastq_r1: str,
    fastq_r2: Optional[str],
    output_dir: str,
    sample_id: str,
    mark_duplicates: bool = True
) -> Dict[str, any]:
    """
    Complete alignment workflow for a sample.
    
    Args:
        fastq_r1: R1 FASTQ path
        fastq_r2: R2 FASTQ path (None for single-end)
        output_dir: Output directory
        sample_id: Sample identifier
        mark_duplicates: Whether to mark duplicate reads
        
    Returns:
        Dictionary with alignment results and metrics
        
    Example:
        >>> results = align_sample(
        ...     "sample_R1.fastq.gz",
        ...     "sample_R2.fastq.gz",
        ...     output_dir="/output/alignment",
        ...     sample_id="SAMPLE001"
        ... )
        >>> print(results["metrics"]["mapping_rate"])
        0.95
    """
    engine = AlignmentEngine()
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Align reads
    raw_bam = str(output_dir_path / f"{sample_id}.raw.bam")
    
    if fastq_r2:
        metrics = engine.align_paired_end(fastq_r1, fastq_r2, raw_bam, sample_id)
    else:
        metrics = engine.align_single_end(fastq_r1, raw_bam, sample_id)
    
    results = {
        "sample_id": sample_id,
        "raw_bam": raw_bam,
        "metrics": metrics.to_dict(),
    }
    
    # Sort and index
    sorted_bam = str(output_dir_path / f"{sample_id}.sorted.bam")
    engine.sort_and_index_bam(raw_bam, sorted_bam)
    results["sorted_bam"] = sorted_bam
    
    # Mark duplicates if requested
    if mark_duplicates:
        dedup_bam = str(output_dir_path / f"{sample_id}.dedup.bam")
        metrics_file = str(output_dir_path / f"{sample_id}.dup_metrics.json")
        dup_stats = engine.mark_duplicates(sorted_bam, dedup_bam, metrics_file)
        results["dedup_bam"] = dedup_bam
        results["duplicate_metrics"] = dup_stats
        results["final_bam"] = dedup_bam
    else:
        results["final_bam"] = sorted_bam
    
    # Calculate coverage
    coverage = engine.calculate_coverage(results["final_bam"])
    results["coverage"] = coverage
    
    # Validate
    is_valid, message = engine.validate_bam(results["final_bam"])
    results["validation"] = {"is_valid": is_valid, "message": message}
    
    return results
