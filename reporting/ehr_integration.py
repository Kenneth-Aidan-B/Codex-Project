"""EHR integration using HL7 FHIR."""

from typing import Dict
import json


def convert_to_fhir(analysis_results: Dict) -> Dict:
    """
    Convert analysis results to HL7 FHIR format.
    
    Args:
        analysis_results: Analysis results dictionary
        
    Returns:
        FHIR-formatted dictionary
    """
    fhir_resource = {
        "resourceType": "DiagnosticReport",
        "status": "final",
        "category": [{
            "coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                "code": "GE",
                "display": "Genetics"
            }]
        }],
        "code": {
            "coding": [{
                "system": "http://loinc.org",
                "code": "81247-9",
                "display": "Master HL7 genetic variant reporting panel"
            }]
        },
        "subject": {
            "reference": f"Patient/{analysis_results.get('sample_id')}"
        },
        "conclusion": f"Risk category: {analysis_results.get('risk_category')}",
        "conclusionCode": [{
            "coding": [{
                "code": analysis_results.get('risk_category'),
                "display": f"Hearing loss risk: {analysis_results.get('risk_category')}"
            }]
        }]
    }
    
    return fhir_resource
