"""
Hearing loss gene database module.

This module provides a comprehensive database of genes associated with
hearing loss, including their clinical significance, inheritance patterns,
and phenotypic information.
"""

from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum


class InheritancePattern(Enum):
    """Inheritance patterns for hearing loss genes."""
    AUTOSOMAL_DOMINANT = "autosomal_dominant"
    AUTOSOMAL_RECESSIVE = "autosomal_recessive"
    X_LINKED = "x_linked"
    MITOCHONDRIAL = "mitochondrial"
    DIGENIC = "digenic"


class HearingLossType(Enum):
    """Types of hearing loss."""
    SENSORINEURAL = "sensorineural"
    CONDUCTIVE = "conductive"
    MIXED = "mixed"


class OnsetAge(Enum):
    """Age of hearing loss onset."""
    CONGENITAL = "congenital"
    EARLY_CHILDHOOD = "early_childhood"
    LATE_CHILDHOOD = "late_childhood"
    ADULT = "adult"
    PROGRESSIVE = "progressive"


@dataclass
class HearingLossGene:
    """Information about a hearing loss gene."""
    
    symbol: str
    name: str
    chromosome: str
    omim_id: str
    dfn_locus: str = ""  # DFNA/DFNB designation
    inheritance: List[InheritancePattern] = field(default_factory=list)
    hearing_loss_type: HearingLossType = HearingLossType.SENSORINEURAL
    onset_age: List[OnsetAge] = field(default_factory=list)
    syndromic: bool = False
    associated_syndromes: List[str] = field(default_factory=list)
    prevalence: str = ""
    clinical_features: str = ""
    molecular_function: str = ""
    
    def to_dict(self) -> Dict:
        """Convert gene info to dictionary."""
        return {
            "symbol": self.symbol,
            "name": self.name,
            "chromosome": self.chromosome,
            "omim_id": self.omim_id,
            "dfn_locus": self.dfn_locus,
            "inheritance": [i.value for i in self.inheritance],
            "hearing_loss_type": self.hearing_loss_type.value,
            "onset_age": [o.value for o in self.onset_age],
            "syndromic": self.syndromic,
            "associated_syndromes": self.associated_syndromes,
            "prevalence": self.prevalence,
            "clinical_features": self.clinical_features,
            "molecular_function": self.molecular_function,
        }


class HearingLossGeneDatabase:
    """Database of hearing loss genes with clinical information."""
    
    def __init__(self):
        """Initialize the gene database."""
        self.genes: Dict[str, HearingLossGene] = {}
        self._initialize_database()
    
    def _initialize_database(self) -> None:
        """Initialize database with known hearing loss genes."""
        
        # Most common genes (top 20)
        self.add_gene(HearingLossGene(
            symbol="GJB2",
            name="Gap Junction Protein Beta 2 (Connexin 26)",
            chromosome="13q12.11",
            omim_id="121011",
            dfn_locus="DFNB1/DFNA3",
            inheritance=[InheritancePattern.AUTOSOMAL_RECESSIVE, InheritancePattern.AUTOSOMAL_DOMINANT],
            hearing_loss_type=HearingLossType.SENSORINEURAL,
            onset_age=[OnsetAge.CONGENITAL],
            syndromic=False,
            prevalence="Most common cause of genetic hearing loss (~50% in some populations)",
            clinical_features="Bilateral sensorineural hearing loss, usually severe to profound",
            molecular_function="Gap junction protein essential for cochlear homeostasis"
        ))
        
        self.add_gene(HearingLossGene(
            symbol="SLC26A4",
            name="Solute Carrier Family 26 Member 4 (Pendrin)",
            chromosome="7q31.1",
            omim_id="605646",
            dfn_locus="DFNB4",
            inheritance=[InheritancePattern.AUTOSOMAL_RECESSIVE],
            hearing_loss_type=HearingLossType.SENSORINEURAL,
            onset_age=[OnsetAge.CONGENITAL, OnsetAge.EARLY_CHILDHOOD],
            syndromic=True,
            associated_syndromes=["Pendred syndrome", "Enlarged vestibular aqueduct (EVA)"],
            prevalence="Second most common cause in many populations",
            clinical_features="Hearing loss with enlarged vestibular aqueduct, may have goiter",
            molecular_function="Chloride-iodide transporter in inner ear and thyroid"
        ))
        
        self.add_gene(HearingLossGene(
            symbol="OTOF",
            name="Otoferlin",
            chromosome="2p23.3",
            omim_id="603681",
            dfn_locus="DFNB9",
            inheritance=[InheritancePattern.AUTOSOMAL_RECESSIVE],
            hearing_loss_type=HearingLossType.SENSORINEURAL,
            onset_age=[OnsetAge.CONGENITAL],
            syndromic=False,
            clinical_features="Auditory neuropathy spectrum disorder (ANSD), absent ABR with present OAEs",
            molecular_function="Synaptic vesicle fusion protein in inner hair cells"
        ))
        
        self.add_gene(HearingLossGene(
            symbol="MYO7A",
            name="Myosin VIIA",
            chromosome="11q13.5",
            omim_id="276903",
            dfn_locus="DFNB2/DFNA11",
            inheritance=[InheritancePattern.AUTOSOMAL_RECESSIVE, InheritancePattern.AUTOSOMAL_DOMINANT],
            hearing_loss_type=HearingLossType.SENSORINEURAL,
            onset_age=[OnsetAge.CONGENITAL],
            syndromic=True,
            associated_syndromes=["Usher syndrome type 1B"],
            clinical_features="Hearing loss with retinitis pigmentosa and vestibular dysfunction",
            molecular_function="Unconventional myosin for hair cell stereocilia"
        ))
        
        self.add_gene(HearingLossGene(
            symbol="CDH23",
            name="Cadherin 23",
            chromosome="10q22.1",
            omim_id="601067",
            dfn_locus="DFNB12",
            inheritance=[InheritancePattern.AUTOSOMAL_RECESSIVE, InheritancePattern.AUTOSOMAL_DOMINANT],
            hearing_loss_type=HearingLossType.SENSORINEURAL,
            onset_age=[OnsetAge.CONGENITAL],
            syndromic=True,
            associated_syndromes=["Usher syndrome type 1D"],
            clinical_features="Profound congenital deafness, may have retinitis pigmentosa",
            molecular_function="Hair cell tip link protein"
        ))
        
        self.add_gene(HearingLossGene(
            symbol="TMC1",
            name="Transmembrane Channel-Like 1",
            chromosome="9q21.13",
            omim_id="606706",
            dfn_locus="DFNB7/DFNA36",
            inheritance=[InheritancePattern.AUTOSOMAL_RECESSIVE, InheritancePattern.AUTOSOMAL_DOMINANT],
            hearing_loss_type=HearingLossType.SENSORINEURAL,
            onset_age=[OnsetAge.CONGENITAL, OnsetAge.PROGRESSIVE],
            clinical_features="Progressive hearing loss, age-related worsening",
            molecular_function="Mechanotransduction channel component"
        ))
        
        self.add_gene(HearingLossGene(
            symbol="TECTA",
            name="Tectorin Alpha",
            chromosome="11q23.3",
            omim_id="602574",
            dfn_locus="DFNA8/12/DFNB21",
            inheritance=[InheritancePattern.AUTOSOMAL_DOMINANT, InheritancePattern.AUTOSOMAL_RECESSIVE],
            hearing_loss_type=HearingLossType.SENSORINEURAL,
            onset_age=[OnsetAge.CONGENITAL, OnsetAge.PROGRESSIVE],
            clinical_features="Mid-frequency hearing loss (characteristic U-shaped audiogram)",
            molecular_function="Tectorial membrane component"
        ))
        
        self.add_gene(HearingLossGene(
            symbol="COCH",
            name="Cochlin",
            chromosome="14q12",
            omim_id="603196",
            dfn_locus="DFNA9",
            inheritance=[InheritancePattern.AUTOSOMAL_DOMINANT],
            hearing_loss_type=HearingLossType.SENSORINEURAL,
            onset_age=[OnsetAge.ADULT, OnsetAge.PROGRESSIVE],
            clinical_features="Progressive hearing loss with vestibular dysfunction",
            molecular_function="Extracellular matrix protein in cochlea"
        ))
        
        self.add_gene(HearingLossGene(
            symbol="MYO15A",
            name="Myosin XVA",
            chromosome="17p11.2",
            omim_id="602666",
            dfn_locus="DFNB3",
            inheritance=[InheritancePattern.AUTOSOMAL_RECESSIVE],
            hearing_loss_type=HearingLossType.SENSORINEURAL,
            onset_age=[OnsetAge.CONGENITAL],
            clinical_features="Congenital profound deafness",
            molecular_function="Required for stereocilia elongation"
        ))
        
        self.add_gene(HearingLossGene(
            symbol="PCDH15",
            name="Protocadherin 15",
            chromosome="10q21.1",
            omim_id="605514",
            dfn_locus="DFNB23",
            inheritance=[InheritancePattern.AUTOSOMAL_RECESSIVE],
            hearing_loss_type=HearingLossType.SENSORINEURAL,
            onset_age=[OnsetAge.CONGENITAL],
            syndromic=True,
            associated_syndromes=["Usher syndrome type 1F"],
            clinical_features="Profound deafness, may have vision and balance problems",
            molecular_function="Hair cell tip link component"
        ))
        
        # Additional important genes
        additional_genes = [
            ("STRC", "Stereocilin", "15q15.3", "606440", "DFNB16"),
            ("ESPN", "Espin", "1p36.31", "606351", "DFNB36"),
            ("MYO6", "Myosin VI", "6q14.1", "600970", "DFNB37/DFNA22"),
            ("ACTG1", "Actin Gamma 1", "17q25.3", "102560", "DFNA20/26"),
            ("POU3F4", "POU Class 3 Homeobox 4", "Xq21.1", "300039", "DFN3"),
            ("COL11A2", "Collagen Type XI Alpha 2", "6p21.32", "120290", "DFNA13"),
            ("TMPRSS3", "Transmembrane Serine Protease 3", "21q22.3", "605511", "DFNB8/10"),
            ("LOXHD1", "Lipoxygenase Homology Domains 1", "18q21.1", "613072", "DFNB77"),
            ("OTOA", "Otoancorin", "7p15.1", "607038", "DFNB22"),
            ("OTOG", "Otogelin", "11p15.3", "604487", "DFNB18B"),
            ("OTOGL", "Otogelin Like", "12q21.2", "614925", "DFNB84B"),
            ("TJP2", "Tight Junction Protein 2", "9q21.11", "607709", "DFNA51"),
            ("TRIOBP", "TRIO and F-Actin Binding Protein", "22q13.1", "609761", "DFNB28"),
            ("USH1C", "Usher Syndrome 1C", "11p15.1", "605242", "DFNB18"),
            ("USH1G", "Usher Syndrome 1G", "17q25.1", "607696", "DFNB18F"),
            ("USH2A", "Usher Syndrome 2A", "1q41", "608400", "USH2A"),
            ("DFNB59", "Pejvakin", "2q31.1", "610220", "DFNB59"),
            ("PJVK", "Pejvakin", "2q31.1", "610219", "DFNB59"),
            ("MARVELD2", "MARVEL Domain Containing 2", "5q13.2", "610572", "DFNB49"),
            ("LHFPL5", "LHFPL Tetraspan Subfamily Member 5", "6p21.31", "609427", "DFNB67"),
        ]
        
        for symbol, name, chrom, omim, dfn in additional_genes:
            self.add_gene(HearingLossGene(
                symbol=symbol,
                name=name,
                chromosome=chrom,
                omim_id=omim,
                dfn_locus=dfn,
                inheritance=[InheritancePattern.AUTOSOMAL_RECESSIVE],
                hearing_loss_type=HearingLossType.SENSORINEURAL,
                onset_age=[OnsetAge.CONGENITAL]
            ))
    
    def add_gene(self, gene: HearingLossGene) -> None:
        """Add a gene to the database."""
        self.genes[gene.symbol] = gene
    
    def get_gene(self, symbol: str) -> Optional[HearingLossGene]:
        """
        Get gene information by symbol.
        
        Args:
            symbol: Gene symbol
            
        Returns:
            HearingLossGene object or None if not found
        """
        return self.genes.get(symbol)
    
    def is_hearing_loss_gene(self, symbol: str) -> bool:
        """
        Check if a gene is associated with hearing loss.
        
        Args:
            symbol: Gene symbol
            
        Returns:
            True if gene is in database
        """
        return symbol in self.genes
    
    def get_all_genes(self) -> List[str]:
        """Get list of all gene symbols."""
        return list(self.genes.keys())
    
    def get_genes_by_inheritance(
        self,
        pattern: InheritancePattern
    ) -> List[HearingLossGene]:
        """Get genes by inheritance pattern."""
        return [
            gene for gene in self.genes.values()
            if pattern in gene.inheritance
        ]
    
    def get_syndromic_genes(self) -> List[HearingLossGene]:
        """Get genes associated with syndromic hearing loss."""
        return [gene for gene in self.genes.values() if gene.syndromic]
    
    def get_nonsyndromic_genes(self) -> List[HearingLossGene]:
        """Get genes associated with nonsyndromic hearing loss."""
        return [gene for gene in self.genes.values() if not gene.syndromic]
    
    def search_genes(self, query: str) -> List[HearingLossGene]:
        """
        Search genes by symbol, name, or syndrome.
        
        Args:
            query: Search query string
            
        Returns:
            List of matching genes
        """
        query_lower = query.lower()
        matches = []
        
        for gene in self.genes.values():
            if (query_lower in gene.symbol.lower() or
                query_lower in gene.name.lower() or
                any(query_lower in s.lower() for s in gene.associated_syndromes)):
                matches.append(gene)
        
        return matches
    
    def get_database_stats(self) -> Dict:
        """Get statistics about the database."""
        total_genes = len(self.genes)
        syndromic = len([g for g in self.genes.values() if g.syndromic])
        
        return {
            "total_genes": total_genes,
            "syndromic_genes": syndromic,
            "nonsyndromic_genes": total_genes - syndromic,
            "autosomal_recessive": len(self.get_genes_by_inheritance(InheritancePattern.AUTOSOMAL_RECESSIVE)),
            "autosomal_dominant": len(self.get_genes_by_inheritance(InheritancePattern.AUTOSOMAL_DOMINANT)),
            "x_linked": len(self.get_genes_by_inheritance(InheritancePattern.X_LINKED)),
        }


# Global database instance
_database: Optional[HearingLossGeneDatabase] = None


def get_gene_database() -> HearingLossGeneDatabase:
    """
    Get the global hearing loss gene database instance.
    
    Returns:
        HearingLossGeneDatabase instance
    """
    global _database
    if _database is None:
        _database = HearingLossGeneDatabase()
    return _database


def get_gene_info(gene_symbol: str) -> Optional[Dict]:
    """
    Get information about a hearing loss gene.
    
    Args:
        gene_symbol: Gene symbol
        
    Returns:
        Dictionary with gene information or None
        
    Example:
        >>> info = get_gene_info("GJB2")
        >>> print(info["name"])
        Gap Junction Protein Beta 2 (Connexin 26)
    """
    db = get_gene_database()
    gene = db.get_gene(gene_symbol)
    return gene.to_dict() if gene else None


def is_hearing_loss_gene(gene_symbol: str) -> bool:
    """
    Check if a gene is associated with hearing loss.
    
    Args:
        gene_symbol: Gene symbol
        
    Returns:
        True if gene is associated with hearing loss
        
    Example:
        >>> is_hearing_loss_gene("GJB2")
        True
        >>> is_hearing_loss_gene("TP53")
        False
    """
    db = get_gene_database()
    return db.is_hearing_loss_gene(gene_symbol)


def get_hearing_loss_genes() -> List[str]:
    """
    Get list of all hearing loss gene symbols.
    
    Returns:
        List of gene symbols
    """
    db = get_gene_database()
    return db.get_all_genes()
