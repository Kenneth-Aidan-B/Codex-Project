"""
VCF file parsing and manipulation module.

This module provides utilities for parsing, querying, and manipulating
VCF (Variant Call Format) files.
"""

from pathlib import Path
from typing import Dict, List, Optional, Iterator, Tuple
from dataclasses import dataclass, field
import gzip


@dataclass
class VCFVariant:
    """Represents a variant from a VCF file."""
    
    chrom: str
    pos: int
    id: str
    ref: str
    alt: str
    qual: Optional[float]
    filter: str
    info: Dict[str, str] = field(default_factory=dict)
    format: Optional[str] = None
    samples: Dict[str, Dict[str, str]] = field(default_factory=dict)
    
    def __str__(self) -> str:
        """String representation of variant."""
        return f"{self.chrom}:{self.pos}:{self.ref}>{self.alt}"
    
    def to_dict(self) -> Dict:
        """Convert variant to dictionary."""
        return {
            "chrom": self.chrom,
            "pos": self.pos,
            "id": self.id,
            "ref": self.ref,
            "alt": self.alt,
            "qual": self.qual,
            "filter": self.filter,
            "info": self.info,
            "format": self.format,
            "samples": self.samples,
        }
    
    def get_genotype(self, sample_id: str) -> Optional[str]:
        """
        Get genotype for a sample.
        
        Args:
            sample_id: Sample identifier
            
        Returns:
            Genotype string (e.g., "0/1") or None
        """
        if sample_id in self.samples and "GT" in self.samples[sample_id]:
            return self.samples[sample_id]["GT"]
        return None
    
    def is_heterozygous(self, sample_id: str) -> bool:
        """Check if variant is heterozygous in sample."""
        gt = self.get_genotype(sample_id)
        return gt in ["0/1", "1/0", "0|1", "1|0"] if gt else False
    
    def is_homozygous_alt(self, sample_id: str) -> bool:
        """Check if variant is homozygous alternate in sample."""
        gt = self.get_genotype(sample_id)
        return gt in ["1/1", "1|1"] if gt else False
    
    def get_allele_depth(self, sample_id: str) -> Tuple[int, int]:
        """
        Get allele depths (ref, alt) for a sample.
        
        Args:
            sample_id: Sample identifier
            
        Returns:
            Tuple of (ref_depth, alt_depth)
        """
        if sample_id in self.samples and "AD" in self.samples[sample_id]:
            ad = self.samples[sample_id]["AD"]
            depths = ad.split(',')
            if len(depths) >= 2:
                return int(depths[0]), int(depths[1])
        return 0, 0
    
    def get_depth(self, sample_id: str) -> int:
        """Get total depth for a sample."""
        if sample_id in self.samples and "DP" in self.samples[sample_id]:
            return int(self.samples[sample_id]["DP"])
        return 0
    
    def get_quality(self, sample_id: str) -> Optional[int]:
        """Get genotype quality for a sample."""
        if sample_id in self.samples and "GQ" in self.samples[sample_id]:
            return int(self.samples[sample_id]["GQ"])
        return None


class VCFParser:
    """Parse and manipulate VCF files."""
    
    def __init__(self, vcf_path: str):
        """
        Initialize VCF parser.
        
        Args:
            vcf_path: Path to VCF file
            
        Raises:
            FileNotFoundError: If VCF file doesn't exist
        """
        if not Path(vcf_path).exists():
            raise FileNotFoundError(f"VCF file not found: {vcf_path}")
        
        self.vcf_path = vcf_path
        self.header_lines: List[str] = []
        self.sample_names: List[str] = []
        self.info_fields: Dict[str, Dict] = {}
        self.format_fields: Dict[str, Dict] = {}
        
        self._parse_header()
    
    def _parse_header(self) -> None:
        """Parse VCF header to extract metadata."""
        open_func = gzip.open if self.vcf_path.endswith('.gz') else open
        
        with open_func(self.vcf_path, 'rt') as f:
            for line in f:
                if line.startswith('##'):
                    self.header_lines.append(line.strip())
                    self._parse_header_line(line.strip())
                elif line.startswith('#CHROM'):
                    # Column header line
                    cols = line.strip().split('\t')
                    if len(cols) > 9:
                        self.sample_names = cols[9:]
                    break
    
    def _parse_header_line(self, line: str) -> None:
        """Parse individual header line for INFO/FORMAT definitions."""
        if line.startswith('##INFO='):
            # Parse INFO field definition
            self._parse_field_definition(line, 'INFO', self.info_fields)
        elif line.startswith('##FORMAT='):
            # Parse FORMAT field definition
            self._parse_field_definition(line, 'FORMAT', self.format_fields)
    
    def _parse_field_definition(
        self,
        line: str,
        field_type: str,
        storage: Dict
    ) -> None:
        """Parse field definition from header."""
        # Simple parser for <ID=...,Number=...,Type=...,Description="...">
        if '<' in line and '>' in line:
            content = line[line.index('<')+1:line.index('>')]
            parts = content.split(',')
            
            field_info = {}
            for part in parts:
                if '=' in part:
                    key, value = part.split('=', 1)
                    field_info[key] = value.strip('"')
            
            if 'ID' in field_info:
                storage[field_info['ID']] = field_info
    
    def parse_variants(self) -> Iterator[VCFVariant]:
        """
        Parse variants from VCF file.
        
        Yields:
            VCFVariant objects
        """
        open_func = gzip.open if self.vcf_path.endswith('.gz') else open
        
        with open_func(self.vcf_path, 'rt') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                
                variant = self._parse_variant_line(line.strip())
                if variant:
                    yield variant
    
    def _parse_variant_line(self, line: str) -> Optional[VCFVariant]:
        """Parse a single variant line."""
        fields = line.split('\t')
        
        if len(fields) < 8:
            return None
        
        # Parse required fields
        chrom = fields[0]
        pos = int(fields[1])
        id_field = fields[2] if fields[2] != '.' else ''
        ref = fields[3]
        alt = fields[4]
        
        # Parse QUAL (can be missing)
        try:
            qual = float(fields[5]) if fields[5] != '.' else None
        except ValueError:
            qual = None
        
        filter_field = fields[6]
        
        # Parse INFO field
        info = self._parse_info_field(fields[7])
        
        # Parse FORMAT and sample data
        format_field = None
        samples = {}
        
        if len(fields) > 8:
            format_field = fields[8]
            format_keys = format_field.split(':')
            
            for i, sample_name in enumerate(self.sample_names):
                if len(fields) > 9 + i:
                    sample_data = fields[9 + i]
                    samples[sample_name] = self._parse_sample_field(
                        format_keys, sample_data
                    )
        
        return VCFVariant(
            chrom=chrom,
            pos=pos,
            id=id_field,
            ref=ref,
            alt=alt,
            qual=qual,
            filter=filter_field,
            info=info,
            format=format_field,
            samples=samples
        )
    
    def _parse_info_field(self, info_str: str) -> Dict[str, str]:
        """Parse INFO field into dictionary."""
        info = {}
        
        if info_str == '.':
            return info
        
        for item in info_str.split(';'):
            if '=' in item:
                key, value = item.split('=', 1)
                info[key] = value
            else:
                # Flag field (no value)
                info[item] = 'True'
        
        return info
    
    def _parse_sample_field(
        self,
        format_keys: List[str],
        sample_data: str
    ) -> Dict[str, str]:
        """Parse sample genotype field."""
        values = sample_data.split(':')
        return dict(zip(format_keys, values))
    
    def filter_by_gene(self, gene_name: str) -> List[VCFVariant]:
        """
        Filter variants by gene name.
        
        Args:
            gene_name: Gene symbol to filter
            
        Returns:
            List of variants in the specified gene
        """
        variants = []
        
        for variant in self.parse_variants():
            # Check if gene is in INFO field
            if 'GENE' in variant.info and gene_name in variant.info['GENE']:
                variants.append(variant)
        
        return variants
    
    def filter_by_region(
        self,
        chrom: str,
        start: int,
        end: int
    ) -> List[VCFVariant]:
        """
        Filter variants by genomic region.
        
        Args:
            chrom: Chromosome
            start: Start position
            end: End position
            
        Returns:
            List of variants in the region
        """
        variants = []
        
        for variant in self.parse_variants():
            if variant.chrom == chrom and start <= variant.pos <= end:
                variants.append(variant)
        
        return variants
    
    def filter_by_quality(self, min_qual: float) -> List[VCFVariant]:
        """
        Filter variants by quality score.
        
        Args:
            min_qual: Minimum quality score
            
        Returns:
            List of high-quality variants
        """
        variants = []
        
        for variant in self.parse_variants():
            if variant.qual and variant.qual >= min_qual:
                variants.append(variant)
        
        return variants
    
    def filter_pathogenic(self) -> List[VCFVariant]:
        """
        Filter variants marked as pathogenic or likely pathogenic.
        
        Returns:
            List of pathogenic variants
        """
        variants = []
        
        for variant in self.parse_variants():
            if 'CLNSIG' in variant.info:
                sig = variant.info['CLNSIG'].lower()
                if 'pathogenic' in sig and 'benign' not in sig:
                    variants.append(variant)
        
        return variants
    
    def filter_hearing_loss_genes(self) -> List[VCFVariant]:
        """
        Filter variants in hearing loss genes.
        
        Returns:
            List of variants in hearing loss genes
        """
        variants = []
        
        for variant in self.parse_variants():
            if 'HL_GENE' in variant.info:
                variants.append(variant)
        
        return variants
    
    def get_variant_count(self) -> int:
        """Get total number of variants in VCF."""
        return sum(1 for _ in self.parse_variants())
    
    def get_sample_names(self) -> List[str]:
        """Get list of sample names in VCF."""
        return self.sample_names.copy()
    
    def to_dataframe(self) -> 'pd.DataFrame':
        """
        Convert VCF to pandas DataFrame.
        
        Returns:
            DataFrame with variant information
            
        Raises:
            ImportError: If pandas not installed
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required for to_dataframe()")
        
        records = []
        
        for variant in self.parse_variants():
            record = {
                'chrom': variant.chrom,
                'pos': variant.pos,
                'id': variant.id,
                'ref': variant.ref,
                'alt': variant.alt,
                'qual': variant.qual,
                'filter': variant.filter,
            }
            
            # Add INFO fields
            for key, value in variant.info.items():
                record[f'info_{key}'] = value
            
            # Add sample genotypes
            for sample_name, sample_data in variant.samples.items():
                for key, value in sample_data.items():
                    record[f'{sample_name}_{key}'] = value
            
            records.append(record)
        
        return pd.DataFrame(records)


def extract_hearing_loss_variants(
    vcf_path: str,
    output_path: Optional[str] = None
) -> List[VCFVariant]:
    """
    Extract variants in hearing loss genes from VCF.
    
    Args:
        vcf_path: Input VCF file path
        output_path: Optional output VCF path for filtered variants
        
    Returns:
        List of VCFVariant objects in hearing loss genes
        
    Example:
        >>> variants = extract_hearing_loss_variants("sample.vcf")
        >>> print(f"Found {len(variants)} hearing loss variants")
        Found 5 hearing loss variants
    """
    parser = VCFParser(vcf_path)
    variants = parser.filter_hearing_loss_genes()
    
    # Write filtered VCF if output path provided
    if output_path:
        write_filtered_vcf(vcf_path, output_path, variants)
    
    return variants


def write_filtered_vcf(
    input_vcf: str,
    output_vcf: str,
    variants: List[VCFVariant]
) -> None:
    """
    Write filtered variants to new VCF file.
    
    Args:
        input_vcf: Input VCF path (for header)
        output_vcf: Output VCF path
        variants: List of variants to write
    """
    parser = VCFParser(input_vcf)
    
    Path(output_vcf).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_vcf, 'w') as f:
        # Write header
        for line in parser.header_lines:
            f.write(f"{line}\n")
        
        # Write column headers
        cols = ['#CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO']
        if parser.sample_names:
            cols.extend(['FORMAT'] + parser.sample_names)
        f.write('\t'.join(cols) + '\n')
        
        # Write variants
        for variant in variants:
            # Build INFO string
            info_parts = []
            for key, value in variant.info.items():
                if value == 'True':
                    info_parts.append(key)
                else:
                    info_parts.append(f"{key}={value}")
            info_str = ';'.join(info_parts) if info_parts else '.'
            
            # Build line
            line_parts = [
                variant.chrom,
                str(variant.pos),
                variant.id or '.',
                variant.ref,
                variant.alt,
                str(variant.qual) if variant.qual else '.',
                variant.filter,
                info_str,
            ]
            
            # Add sample data if present
            if variant.format and variant.samples:
                line_parts.append(variant.format)
                for sample_name in parser.sample_names:
                    if sample_name in variant.samples:
                        format_keys = variant.format.split(':')
                        sample_values = [
                            variant.samples[sample_name].get(k, '.')
                            for k in format_keys
                        ]
                        line_parts.append(':'.join(sample_values))
            
            f.write('\t'.join(line_parts) + '\n')


def compare_vcfs(vcf1_path: str, vcf2_path: str) -> Dict[str, any]:
    """
    Compare two VCF files and report differences.
    
    Args:
        vcf1_path: First VCF file
        vcf2_path: Second VCF file
        
    Returns:
        Dictionary with comparison statistics
        
    Example:
        >>> stats = compare_vcfs("sample1.vcf", "sample2.vcf")
        >>> print(f"Shared variants: {stats['shared']}")
    """
    parser1 = VCFParser(vcf1_path)
    parser2 = VCFParser(vcf2_path)
    
    # Get variant positions
    variants1 = {(v.chrom, v.pos, v.ref, v.alt) for v in parser1.parse_variants()}
    variants2 = {(v.chrom, v.pos, v.ref, v.alt) for v in parser2.parse_variants()}
    
    shared = variants1 & variants2
    unique1 = variants1 - variants2
    unique2 = variants2 - variants1
    
    return {
        "vcf1_variants": len(variants1),
        "vcf2_variants": len(variants2),
        "shared": len(shared),
        "unique_to_vcf1": len(unique1),
        "unique_to_vcf2": len(unique2),
        "jaccard_index": len(shared) / len(variants1 | variants2) if (variants1 | variants2) else 0,
    }
