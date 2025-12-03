"""Tests for pipeline configuration."""

import pytest
from pipeline.config import get_config, PipelineConfig


def test_get_config():
    """Test getting default configuration."""
    config = get_config()
    assert isinstance(config, PipelineConfig)
    assert config.reference.name == "hg38"


def test_hearing_loss_genes():
    """Test hearing loss gene configuration."""
    config = get_config()
    genes = config.hearing_genes.get_all_genes()
    
    assert "GJB2" in genes
    assert "SLC26A4" in genes
    assert "OTOF" in genes
    assert len(genes) > 10


def test_primary_genes():
    """Test primary hearing loss genes."""
    config = get_config()
    primary = config.hearing_genes.primary_genes
    
    assert "GJB2" in primary
    assert "MYO7A" in primary
