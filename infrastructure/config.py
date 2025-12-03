"""Infrastructure configuration."""

import os
from dataclasses import dataclass


@dataclass
class InfrastructureConfig:
    """Infrastructure settings."""
    
    # Cloud storage
    s3_bucket: str = os.getenv("S3_BUCKET", "")
    s3_region: str = os.getenv("S3_REGION", "us-east-1")
    
    # Encryption
    encryption_enabled: bool = os.getenv("ENCRYPTION_ENABLED", "true").lower() == "true"
    encryption_key_id: str = os.getenv("ENCRYPTION_KEY_ID", "")
    
    # API
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    
    # Database
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./hearing_screening.db")
    
    # Monitoring
    enable_metrics: bool = os.getenv("ENABLE_METRICS", "true").lower() == "true"
    metrics_port: int = int(os.getenv("METRICS_PORT", "9090"))


def get_infrastructure_config() -> InfrastructureConfig:
    """Get infrastructure configuration."""
    return InfrastructureConfig()
