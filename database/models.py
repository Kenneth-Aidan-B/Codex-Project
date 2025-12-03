"""SQLAlchemy ORM models for database."""

from datetime import datetime
from typing import Optional
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class Sample(Base):
    """Sample information."""
    __tablename__ = 'samples'
    
    id = Column(Integer, primary_key=True)
    external_id = Column(String(100), unique=True, nullable=False, index=True)
    birth_date = Column(DateTime)
    sex = Column(String(10))
    ethnicity = Column(String(100))
    collection_date = Column(DateTime)
    sequencing_date = Column(DateTime)
    status = Column(String(50), default='received')
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    analyses = relationship("Analysis", back_populates="sample")


class Analysis(Base):
    """Analysis results."""
    __tablename__ = 'analyses'
    
    id = Column(Integer, primary_key=True)
    sample_id = Column(Integer, ForeignKey('samples.id'), nullable=False)
    model_version = Column(String(50))
    risk_score = Column(Float)
    risk_category = Column(String(20))
    confidence = Column(Float)
    status = Column(String(50), default='pending')
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    results_json = Column(JSON)
    
    # Relationships
    sample = relationship("Sample", back_populates="analyses")
    variants = relationship("Variant", back_populates="analysis")
    reports = relationship("Report", back_populates="analysis")


class Variant(Base):
    """Variant information."""
    __tablename__ = 'variants'
    
    id = Column(Integer, primary_key=True)
    analysis_id = Column(Integer, ForeignKey('analyses.id'), nullable=False)
    chromosome = Column(String(10), nullable=False)
    position = Column(Integer, nullable=False)
    ref = Column(String(1000))
    alt = Column(String(1000))
    gene = Column(String(50), index=True)
    consequence = Column(String(100))
    clinvar_sig = Column(String(100))
    gnomad_af = Column(Float)
    pathogenicity_score = Column(Float)
    
    # Relationships
    analysis = relationship("Analysis", back_populates="variants")


class Report(Base):
    """Generated reports."""
    __tablename__ = 'reports'
    
    id = Column(Integer, primary_key=True)
    analysis_id = Column(Integer, ForeignKey('analyses.id'), nullable=False)
    format = Column(String(20))  # 'pdf', 'json', 'fhir'
    content = Column(Text)
    file_path = Column(String(500))
    generated_at = Column(DateTime, default=datetime.utcnow)
    recipient_id = Column(String(100))
    sent_at = Column(DateTime)
    
    # Relationships
    analysis = relationship("Analysis", back_populates="reports")


class AuditLog(Base):
    """Audit log for compliance."""
    __tablename__ = 'audit_logs'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String(100))
    action = Column(String(100), nullable=False)
    resource = Column(String(100))
    resource_id = Column(String(100))
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    ip_address = Column(String(50))
    details = Column(JSON)
