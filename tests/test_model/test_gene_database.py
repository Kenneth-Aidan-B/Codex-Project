"""Tests for hearing loss gene database."""

import pytest
from model.gene_database import (
    get_gene_database,
    get_gene_info,
    is_hearing_loss_gene,
    get_hearing_loss_genes
)


def test_get_gene_database():
    """Test getting gene database instance."""
    db = get_gene_database()
    assert db is not None
    assert len(db.genes) > 0


def test_gjb2_gene():
    """Test GJB2 (most common hearing loss gene)."""
    info = get_gene_info("GJB2")
    assert info is not None
    assert info["symbol"] == "GJB2"
    assert "Connexin" in info["name"]
    assert info["chromosome"] == "13q12.11"


def test_slc26a4_gene():
    """Test SLC26A4 (Pendred syndrome)."""
    info = get_gene_info("SLC26A4")
    assert info is not None
    assert "Pendrin" in info["name"]
    assert "Pendred syndrome" in str(info["associated_syndromes"])


def test_is_hearing_loss_gene():
    """Test checking if gene is hearing loss related."""
    assert is_hearing_loss_gene("GJB2") == True
    assert is_hearing_loss_gene("SLC26A4") == True
    assert is_hearing_loss_gene("TP53") == False
    assert is_hearing_loss_gene("UNKNOWN_GENE") == False


def test_get_all_genes():
    """Test getting all hearing loss genes."""
    genes = get_hearing_loss_genes()
    assert isinstance(genes, list)
    assert len(genes) > 20  # Should have at least 20 genes
    assert "GJB2" in genes
    assert "OTOF" in genes


def test_database_stats():
    """Test database statistics."""
    db = get_gene_database()
    stats = db.get_database_stats()
    
    assert "total_genes" in stats
    assert stats["total_genes"] > 0
    assert "syndromic_genes" in stats
    assert "autosomal_recessive" in stats
