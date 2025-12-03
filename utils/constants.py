"""Application-wide constants."""

# Risk thresholds
RISK_THRESHOLD_LOW = 0.3
RISK_THRESHOLD_HIGH = 0.7

# Quality thresholds
MIN_VARIANT_QUALITY = 30.0
MIN_GENOTYPE_QUALITY = 20
MIN_READ_DEPTH = 10

# Population frequency thresholds
RARE_VARIANT_AF = 0.01
VERY_RARE_VARIANT_AF = 0.001

# ClinVar significance categories
PATHOGENIC_CATEGORIES = ["pathogenic", "likely_pathogenic"]
BENIGN_CATEGORIES = ["benign", "likely_benign"]

# Chromosomes
AUTOSOMES = [str(i) for i in range(1, 23)]
SEX_CHROMOSOMES = ['X', 'Y']
MITOCHONDRIAL = ['MT', 'M']
ALL_CHROMOSOMES = AUTOSOMES + SEX_CHROMOSOMES + MITOCHONDRIAL
