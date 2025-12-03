"""
Variant calling module for identifying SNPs and indels from aligned reads.

This module provides functionality for calling variants from BAM files
using various calling algorithms.
"""

import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json

from pipeline.config import get_config, VariantCallingConfig, get_tool_path


@dataclass
class VariantCallStats:
    """Statistics from variant calling."""
    
    total_variants: int = 0
    snps: int = 0
    indels: int = 0
    multiallelic: int = 0
    passed_filter: int = 0
    mean_quality: float = 0.0
    mean_depth: float = 0.0
    ti_tv_ratio: float = 0.0  # Transition/transversion ratio
    
    def to_dict(self) -> Dict:
        """Convert stats to dictionary."""
        return {
            "total_variants": self.total_variants,
            "snps": self.snps,
            "indels": self.indels,
            "multiallelic": self.multiallelic,
            "passed_filter": self.passed_filter,
            "pass_rate": round(self.passed_filter / max(self.total_variants, 1), 4),
            "mean_quality": round(self.mean_quality, 2),
            "mean_depth": round(self.mean_depth, 2),
            "ti_tv_ratio": round(self.ti_tv_ratio, 4),
        }


class VariantCaller:
    """Call variants from aligned reads."""
    
    def __init__(self, config: Optional[VariantCallingConfig] = None):
        """
        Initialize variant caller.
        
        Args:
            config: Variant calling configuration (uses default if None)
        """
        self.config = config or get_config().variant_calling
        self.reference_config = get_config().reference
    
    def call_variants(
        self,
        bam_file: str,
        output_vcf: str,
        regions: Optional[str] = None
    ) -> VariantCallStats:
        """
        Call variants from BAM file.
        
        Args:
            bam_file: Input BAM file path
            output_vcf: Output VCF file path
            regions: Optional BED file with regions to call
            
        Returns:
            VariantCallStats object
            
        Raises:
            FileNotFoundError: If BAM or reference not found
            RuntimeError: If variant calling fails
        """
        if not Path(bam_file).exists():
            raise FileNotFoundError(f"BAM file not found: {bam_file}")
        
        if not Path(self.reference_config.fasta_path).exists():
            raise FileNotFoundError(f"Reference not found: {self.reference_config.fasta_path}")
        
        Path(output_vcf).parent.mkdir(parents=True, exist_ok=True)
        
        # Build variant calling command
        cmd = self._build_calling_command(bam_file, output_vcf, regions)
        
        # Execute calling (mock implementation)
        stats = self._execute_calling_mock(cmd, output_vcf)
        
        return stats
    
    def _build_calling_command(
        self,
        bam_file: str,
        output_vcf: str,
        regions: Optional[str]
    ) -> list:
        """
        Build variant calling command using bcftools mpileup + call.
        
        Args:
            bam_file: BAM file path
            output_vcf: Output VCF path
            regions: Optional regions BED file
            
        Returns:
            List of command components
        """
        # bcftools mpileup + call pipeline
        cmd = [
            get_tool_path("bcftools"),
            "mpileup",
            "-f", self.reference_config.fasta_path,
            "-q", str(self.config.min_mapping_quality),
            "-Q", str(self.config.min_base_quality),
            "--max-depth", str(self.config.max_coverage),
            "-a", "FORMAT/AD,FORMAT/DP",
            bam_file,
        ]
        
        if regions:
            cmd.extend(["-R", regions])
        
        return cmd
    
    def _execute_calling_mock(
        self,
        cmd: list,
        output_vcf: str
    ) -> VariantCallStats:
        """
        Execute variant calling (mock implementation).
        
        In production, this would run bcftools/GATK and generate real VCF.
        
        Args:
            cmd: Command list
            output_vcf: Output VCF path
            
        Returns:
            VariantCallStats with mock data
        """
        # Create mock VCF file
        self._create_mock_vcf(output_vcf)
        
        # Generate mock statistics
        stats = VariantCallStats(
            total_variants=1250,
            snps=1000,
            indels=250,
            multiallelic=50,
            passed_filter=1100,
            mean_quality=35.5,
            mean_depth=32.0,
            ti_tv_ratio=2.1,
        )
        
        return stats
    
    def _create_mock_vcf(self, output_vcf: str) -> None:
        """Create a mock VCF file with header and sample variants."""
        vcf_content = """##fileformat=VCFv4.2
##reference=hg38
##contig=<ID=chr1,length=248956422>
##contig=<ID=chr2,length=242193529>
##INFO=<ID=DP,Number=1,Type=Integer,Description="Total Depth">
##INFO=<ID=AF,Number=A,Type=Float,Description="Allele Frequency">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read Depth">
##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allelic Depths">
##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype Quality">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE
chr13\t20763612\t.\tC\tT\t45.2\tPASS\tDP=30;AF=0.5\tGT:DP:AD:GQ\t0/1:30:15,15:45
chr13\t20763686\t.\tG\tA\t52.1\tPASS\tDP=35;AF=0.5\tGT:DP:AD:GQ\t0/1:35:18,17:52
chr7\t117548630\t.\tA\tG\t38.5\tPASS\tDP=28;AF=0.5\tGT:DP:AD:GQ\t0/1:28:14,14:38
"""
        with open(output_vcf, 'w') as f:
            f.write(vcf_content)
        
        # Create index
        Path(f"{output_vcf}.tbi").touch()
    
    def filter_variants(
        self,
        input_vcf: str,
        output_vcf: str,
        min_quality: Optional[float] = None,
        min_depth: Optional[int] = None,
        max_missing: float = 0.1
    ) -> VariantCallStats:
        """
        Filter variants based on quality criteria.
        
        Args:
            input_vcf: Input VCF file
            output_vcf: Output filtered VCF file
            min_quality: Minimum QUAL score (uses config default if None)
            min_depth: Minimum depth (uses config default if None)
            max_missing: Maximum missing genotype rate
            
        Returns:
            VariantCallStats for filtered variants
            
        Raises:
            FileNotFoundError: If input VCF not found
        """
        if not Path(input_vcf).exists():
            raise FileNotFoundError(f"VCF file not found: {input_vcf}")
        
        min_quality = min_quality or self.config.call_confidence_threshold
        min_depth = min_depth or self.config.min_coverage
        
        Path(output_vcf).parent.mkdir(parents=True, exist_ok=True)
        
        # Mock implementation - copy VCF and return stats
        self._create_mock_vcf(output_vcf)
        
        stats = VariantCallStats(
            total_variants=1100,
            snps=900,
            indels=200,
            multiallelic=40,
            passed_filter=1100,
            mean_quality=40.0,
            mean_depth=35.0,
            ti_tv_ratio=2.15,
        )
        
        return stats
    
    def call_variants_gatk(
        self,
        bam_file: str,
        output_vcf: str,
        intervals: Optional[str] = None
    ) -> VariantCallStats:
        """
        Call variants using GATK HaplotypeCaller.
        
        Args:
            bam_file: Input BAM file
            output_vcf: Output VCF file
            intervals: Optional intervals file
            
        Returns:
            VariantCallStats object
            
        Note:
            This is a placeholder. In production, would use actual GATK.
        """
        if not Path(bam_file).exists():
            raise FileNotFoundError(f"BAM file not found: {bam_file}")
        
        Path(output_vcf).parent.mkdir(parents=True, exist_ok=True)
        
        # Mock GATK calling
        self._create_mock_vcf(output_vcf)
        
        stats = VariantCallStats(
            total_variants=1300,
            snps=1050,
            indels=250,
            multiallelic=55,
            passed_filter=1150,
            mean_quality=42.0,
            mean_depth=33.0,
            ti_tv_ratio=2.08,
        )
        
        return stats
    
    def joint_genotyping(
        self,
        gvcf_files: List[str],
        output_vcf: str
    ) -> VariantCallStats:
        """
        Perform joint genotyping on multiple GVCF files.
        
        Args:
            gvcf_files: List of GVCF file paths
            output_vcf: Output joint-called VCF
            
        Returns:
            VariantCallStats object
            
        Raises:
            ValueError: If gvcf_files is empty
            FileNotFoundError: If any GVCF file not found
        """
        if not gvcf_files:
            raise ValueError("No GVCF files provided")
        
        for gvcf in gvcf_files:
            if not Path(gvcf).exists():
                raise FileNotFoundError(f"GVCF not found: {gvcf}")
        
        Path(output_vcf).parent.mkdir(parents=True, exist_ok=True)
        
        # Mock joint genotyping
        self._create_mock_vcf(output_vcf)
        
        stats = VariantCallStats(
            total_variants=len(gvcf_files) * 1200,
            snps=len(gvcf_files) * 950,
            indels=len(gvcf_files) * 250,
            multiallelic=len(gvcf_files) * 50,
            passed_filter=len(gvcf_files) * 1050,
            mean_quality=38.0,
            mean_depth=30.0,
            ti_tv_ratio=2.12,
        )
        
        return stats
    
    def calculate_vcf_stats(self, vcf_file: str) -> VariantCallStats:
        """
        Calculate statistics for a VCF file.
        
        Args:
            vcf_file: VCF file path
            
        Returns:
            VariantCallStats object
            
        Raises:
            FileNotFoundError: If VCF file not found
        """
        if not Path(vcf_file).exists():
            raise FileNotFoundError(f"VCF file not found: {vcf_file}")
        
        # Parse VCF and calculate stats
        total_variants = 0
        snps = 0
        indels = 0
        multiallelic = 0
        passed_filter = 0
        
        with open(vcf_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                
                total_variants += 1
                fields = line.strip().split('\t')
                
                ref = fields[3]
                alt = fields[4]
                filter_field = fields[6]
                
                # Count variant types
                if len(ref) == 1 and len(alt) == 1:
                    snps += 1
                else:
                    indels += 1
                
                if ',' in alt:
                    multiallelic += 1
                
                if filter_field == 'PASS' or filter_field == '.':
                    passed_filter += 1
        
        stats = VariantCallStats(
            total_variants=total_variants,
            snps=snps,
            indels=indels,
            multiallelic=multiallelic,
            passed_filter=passed_filter,
            mean_quality=35.0,  # Would calculate from actual data
            mean_depth=30.0,
            ti_tv_ratio=2.1,
        )
        
        return stats


def call_variants_for_sample(
    bam_file: str,
    output_dir: str,
    sample_id: str,
    regions: Optional[str] = None,
    filter_variants: bool = True
) -> Dict[str, any]:
    """
    Complete variant calling workflow for a sample.
    
    Args:
        bam_file: Input BAM file
        output_dir: Output directory
        sample_id: Sample identifier
        regions: Optional regions BED file
        filter_variants: Whether to apply quality filters
        
    Returns:
        Dictionary with variant calling results
        
    Example:
        >>> results = call_variants_for_sample(
        ...     "sample.bam",
        ...     output_dir="/output/variants",
        ...     sample_id="SAMPLE001"
        ... )
        >>> print(results["stats"]["total_variants"])
        1250
    """
    caller = VariantCaller()
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Call variants
    raw_vcf = str(output_dir_path / f"{sample_id}.raw.vcf")
    raw_stats = caller.call_variants(bam_file, raw_vcf, regions)
    
    results = {
        "sample_id": sample_id,
        "raw_vcf": raw_vcf,
        "raw_stats": raw_stats.to_dict(),
    }
    
    # Filter variants if requested
    if filter_variants:
        filtered_vcf = str(output_dir_path / f"{sample_id}.filtered.vcf")
        filtered_stats = caller.filter_variants(raw_vcf, filtered_vcf)
        results["filtered_vcf"] = filtered_vcf
        results["filtered_stats"] = filtered_stats.to_dict()
        results["final_vcf"] = filtered_vcf
        results["stats"] = filtered_stats.to_dict()
    else:
        results["final_vcf"] = raw_vcf
        results["stats"] = raw_stats.to_dict()
    
    return results
