"""Input validation utilities."""

import re
from typing import Optional


def validate_sample_id(sample_id: str) -> bool:
    """Validate sample ID format."""
    return bool(re.match(r'^[A-Z0-9_-]+$', sample_id))


def validate_vcf_path(path: str) -> bool:
    """Validate VCF file path."""
    return path.endswith(('.vcf', '.vcf.gz'))


def validate_chromosome(chrom: str) -> bool:
    """Validate chromosome identifier."""
    valid_chroms = [str(i) for i in range(1, 23)] + ['X', 'Y', 'MT', 'M']
    valid_chroms += ['chr' + c for c in valid_chroms]
    return chrom in valid_chroms
