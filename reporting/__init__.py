"""Reporting system."""

from reporting.report_generator import generate_clinical_report
from reporting.ehr_integration import convert_to_fhir

__all__ = ["generate_clinical_report", "convert_to_fhir"]
