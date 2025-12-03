# Contributing to AI-Based Genomic Newborn Hearing Screening Platform

Thank you for your interest in contributing to this project! This document provides guidelines and best practices for contributing.

---

## Table of Contents
1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Workflow](#development-workflow)
4. [Coding Standards](#coding-standards)
5. [Testing Requirements](#testing-requirements)
6. [Documentation](#documentation)
7. [Submitting Changes](#submitting-changes)
8. [Review Process](#review-process)

---

## Code of Conduct

This project adheres to our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

---

## Getting Started

### Prerequisites
- Python 3.9+
- Docker & Docker Compose
- Git
- PostgreSQL (for database development)

### Setting Up Development Environment

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-org/Codex-Project.git
   cd Codex-Project
   ```

2. **Create virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

4. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your local settings
   ```

5. **Initialize database** (optional for local development):
   ```bash
   python -m database.migrations.init_db
   ```

6. **Run tests** to verify setup:
   ```bash
   pytest tests/
   ```

---

## Development Workflow

### Branching Strategy

We follow **Git Flow**:

- `main`: Production-ready code
- `develop`: Integration branch for features
- `feature/*`: New features
- `bugfix/*`: Bug fixes
- `hotfix/*`: Critical production fixes
- `release/*`: Release preparation

### Creating a Feature Branch

```bash
git checkout develop
git pull origin develop
git checkout -b feature/my-new-feature
```

### Commit Message Format

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples**:
```
feat(pipeline): add support for hg19 reference genome

Implemented alignment and variant calling for hg19 in addition to hg38.
Added configuration option to select reference genome.

Closes #123
```

```
fix(api): correct risk score calculation for compound heterozygotes

Previously, compound heterozygotes were not properly identified,
leading to underestimated risk scores.

Fixes #456
```

---

## Coding Standards

### Python Style Guide

We follow **PEP 8** with some modifications:

- **Line length**: 100 characters (not 79)
- **Indentation**: 4 spaces
- **Quotes**: Prefer double quotes for strings
- **Imports**: Organized as:
  1. Standard library
  2. Third-party packages
  3. Local modules

### Type Hints

**All public functions must have type hints**:

```python
from typing import List, Dict, Optional

def process_variants(
    vcf_path: str,
    gene_list: List[str],
    min_qual: float = 30.0
) -> Dict[str, List[Variant]]:
    """
    Process variants from VCF file.

    Args:
        vcf_path: Path to VCF file
        gene_list: List of genes to filter
        min_qual: Minimum quality score (default: 30.0)

    Returns:
        Dictionary mapping gene names to variant lists

    Raises:
        FileNotFoundError: If VCF file doesn't exist
        ValueError: If gene_list is empty
    """
    pass
```

### Docstrings

Use **Google-style docstrings**:

```python
def calculate_risk_score(variants: List[Variant], model: Model) -> float:
    """Calculate hearing loss risk score from variants.

    This function applies the trained ML model to genomic variants
    and clinical features to produce a risk score between 0 and 1.

    Args:
        variants: List of annotated variants for the sample
        model: Trained prediction model

    Returns:
        Risk score (0.0 = low risk, 1.0 = high risk)

    Raises:
        ValueError: If variants list is empty
        ModelError: If model inference fails

    Example:
        >>> variants = load_variants("sample123.vcf")
        >>> model = load_model("production_v2.pkl")
        >>> score = calculate_risk_score(variants, model)
        >>> print(f"Risk: {score:.2%}")
        Risk: 23.45%
    """
    pass
```

### Code Organization

```python
# 1. Module docstring
"""Module for variant annotation."""

# 2. Imports
import os
from typing import List, Dict
from pathlib import Path

import pandas as pd
from pysam import VariantFile

from utils.logging import get_logger
from utils.validators import validate_vcf

# 3. Constants
DEFAULT_QUALITY_THRESHOLD = 30.0
SUPPORTED_CHROMOSOMES = [str(i) for i in range(1, 23)] + ["X", "Y", "MT"]

# 4. Logger
logger = get_logger(__name__)

# 5. Classes and functions
class VariantAnnotator:
    """Annotates variants with clinical databases."""
    pass

def annotate_vcf(vcf_path: str) -> pd.DataFrame:
    """Annotate variants in VCF file."""
    pass
```

### Error Handling

- **Use specific exceptions**: Avoid bare `except:`
- **Provide context**: Include meaningful error messages
- **Log errors**: Use structured logging

```python
from utils.logging import get_logger

logger = get_logger(__name__)

try:
    result = risky_operation()
except FileNotFoundError as e:
    logger.error(f"File not found: {e}")
    raise
except ValueError as e:
    logger.error(f"Invalid value: {e}")
    # Handle gracefully or re-raise
    raise
except Exception as e:
    logger.exception(f"Unexpected error: {e}")
    raise RuntimeError(f"Operation failed: {e}") from e
```

### Security Best Practices

1. **Never commit secrets**: Use environment variables
2. **Validate all inputs**: Check file paths, user data
3. **Sanitize outputs**: Prevent injection attacks
4. **Use parameterized queries**: Avoid SQL injection
5. **Encrypt sensitive data**: Use encryption utilities
6. **Log security events**: Track authentication, authorization

---

## Testing Requirements

### Test Coverage

- **Minimum coverage**: 80% per module
- **Critical modules**: 95%+ coverage (pipeline, model, api)

### Test Types

**Unit Tests** (`tests/test_*/`):
```python
import pytest
from pipeline.variant_caller import call_variants

def test_call_variants_basic():
    """Test basic variant calling."""
    result = call_variants("test.bam", "chr1", 1000, 2000)
    assert len(result) > 0
    assert result[0].chrom == "chr1"

def test_call_variants_invalid_input():
    """Test error handling for invalid input."""
    with pytest.raises(FileNotFoundError):
        call_variants("nonexistent.bam", "chr1", 1000, 2000)
```

**Integration Tests** (`tests/test_integration/`):
```python
def test_end_to_end_pipeline(tmp_path):
    """Test complete pipeline from FASTQ to report."""
    # Setup
    fastq = create_test_fastq(tmp_path)
    
    # Execute pipeline
    vcf = run_alignment_and_calling(fastq)
    analysis = run_prediction(vcf)
    report = generate_report(analysis)
    
    # Verify
    assert report.risk_score > 0
    assert len(report.variants) > 0
```

**API Tests** (`tests/test_api/`):
```python
from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)

def test_predict_endpoint():
    """Test prediction API endpoint."""
    payload = {
        "sample_id": "TEST001",
        "variants": [...]
    }
    response = client.post("/api/analysis/predict", json=payload)
    assert response.status_code == 200
    assert "risk_score" in response.json()
```

### Running Tests

```bash
# All tests
pytest

# Specific module
pytest tests/test_pipeline/

# With coverage
pytest --cov=. --cov-report=html

# Parallel execution
pytest -n auto

# Verbose output
pytest -v -s
```

### Mocking External Services

```python
from unittest.mock import Mock, patch

@patch('pipeline.annotator.fetch_clinvar_data')
def test_annotate_with_clinvar(mock_fetch):
    """Test annotation with mocked ClinVar API."""
    mock_fetch.return_value = {"variant_id": "VCV000123", "significance": "Pathogenic"}
    
    result = annotate_variant("chr1", 12345, "A", "G")
    
    assert result["clinvar_significance"] == "Pathogenic"
    mock_fetch.assert_called_once()
```

**Important**: Mock Gemini API calls in tests:
```python
@patch('api.ai_insight.get_gemini_response')
def test_ai_insight(mock_gemini):
    """Test AI insight generation."""
    mock_gemini.return_value = "Mock AI response"
    result = generate_insight(sample_data)
    assert "Mock AI response" in result
```

---

## Documentation

### Code Documentation

- **All public APIs**: Comprehensive docstrings
- **Complex algorithms**: Inline comments explaining logic
- **Configuration**: Document all options

### Project Documentation

When adding new features, update:

1. **README.md**: High-level overview
2. **ARCHITECTURE.md**: System design changes
3. **docs/**: Detailed user guides
4. **API docs**: OpenAPI/Swagger annotations

### API Documentation

Use FastAPI's built-in OpenAPI:

```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

class PredictionRequest(BaseModel):
    """Request model for prediction endpoint."""
    sample_id: str
    variants: List[Dict]

@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict hearing loss risk",
    description="Analyzes genomic variants to predict hearing loss risk score",
    tags=["Analysis"]
)
async def predict(request: PredictionRequest):
    """Predict hearing loss risk from genomic variants."""
    pass
```

---

## Submitting Changes

### Before Submitting

1. **Run tests**: `pytest`
2. **Check code style**: `flake8 . && black --check .`
3. **Type check**: `mypy .`
4. **Update documentation**: If applicable
5. **Rebase on latest develop**: `git rebase develop`

### Pull Request Process

1. **Create PR** against `develop` branch
2. **Fill PR template** with:
   - Description of changes
   - Related issue numbers
   - Testing performed
   - Screenshots (if UI changes)
3. **Request review** from relevant maintainers
4. **Address feedback** promptly
5. **Squash commits** if requested

### PR Title Format

```
<type>(<scope>): <description>

Example:
feat(model): add ensemble voting classifier
fix(api): resolve race condition in batch processing
docs(readme): update installation instructions
```

---

## Review Process

### For Contributors

- Respond to feedback within 2 business days
- Be open to suggestions and constructive criticism
- Ask questions if requirements are unclear

### For Reviewers

Check for:

1. **Correctness**: Does the code work as intended?
2. **Tests**: Are there adequate tests?
3. **Style**: Does it follow coding standards?
4. **Documentation**: Are changes documented?
5. **Security**: Are there security implications?
6. **Performance**: Are there performance concerns?

### Review Checklist

- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] No security vulnerabilities introduced
- [ ] No hardcoded credentials or secrets
- [ ] Error handling is appropriate
- [ ] Logging is adequate

---

## Community

### Communication Channels

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: General questions, ideas
- **Slack**: Real-time collaboration (invite link)
- **Mailing List**: Announcements, releases

### Getting Help

- Check existing documentation
- Search GitHub issues
- Ask in GitHub Discussions
- Reach out to maintainers

---

## Recognition

Contributors will be acknowledged in:
- `CONTRIBUTORS.md` file
- Release notes
- Project documentation

Thank you for contributing to improving newborn hearing screening! 🎉

---

**Document Version**: 1.0  
**Last Updated**: 2025-12-03
