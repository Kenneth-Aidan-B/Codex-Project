"""Tests for FASTQ processing."""

import pytest
from pathlib import Path
from pipeline.fastq_processor import FastqProcessor, ReadQualityMetrics


def test_fastq_processor_init():
    """Test FASTQ processor initialization."""
    processor = FastqProcessor()
    assert processor.config is not None


def test_quality_metrics():
    """Test quality metrics dataclass."""
    metrics = ReadQualityMetrics(
        total_reads=1000,
        total_bases=100000,
        mean_read_length=100.0,
        passed_filter_reads=950
    )
    
    assert metrics.total_reads == 1000
    data = metrics.to_dict()
    assert "total_reads" in data
    assert data["pass_rate"] == 0.95
