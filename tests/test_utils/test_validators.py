"""Tests for validation utilities."""

import pytest
from utils.validators import validate_sample_id, validate_vcf_path, validate_chromosome


def test_validate_sample_id():
    """Test sample ID validation."""
    assert validate_sample_id("SAMPLE001") == True
    assert validate_sample_id("TEST_123") == True
    assert validate_sample_id("SAMPLE-456") == True
    assert validate_sample_id("sample@123") == False  # Invalid character
    assert validate_sample_id("") == False  # Empty


def test_validate_vcf_path():
    """Test VCF path validation."""
    assert validate_vcf_path("sample.vcf") == True
    assert validate_vcf_path("sample.vcf.gz") == True
    assert validate_vcf_path("sample.bam") == False
    assert validate_vcf_path("sample.txt") == False


def test_validate_chromosome():
    """Test chromosome validation."""
    assert validate_chromosome("1") == True
    assert validate_chromosome("22") == True
    assert validate_chromosome("X") == True
    assert validate_chromosome("Y") == True
    assert validate_chromosome("MT") == True
    assert validate_chromosome("chr1") == True
    assert validate_chromosome("chrX") == True
    assert validate_chromosome("25") == False  # Invalid
    assert validate_chromosome("ABC") == False  # Invalid
