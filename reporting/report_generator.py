"""Clinical report generation."""

from typing import Dict, Optional
from datetime import datetime
import json
from pathlib import Path


def generate_clinical_report(
    analysis_results: Dict,
    output_path: str,
    format: str = "json"
) -> str:
    """
    Generate clinical report from analysis results.
    
    Args:
        analysis_results: Analysis results dictionary
        output_path: Output file path
        format: Report format ('json', 'pdf', 'html')
        
    Returns:
        Path to generated report
    """
    report = {
        "report_id": f"RPT{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "generated_at": datetime.now().isoformat(),
        "sample_id": analysis_results.get("sample_id"),
        "risk_assessment": {
            "overall_risk_score": analysis_results.get("risk_score"),
            "risk_category": analysis_results.get("risk_category"),
            "confidence": analysis_results.get("confidence")
        },
        "findings": analysis_results.get("key_findings", []),
        "top_genes": analysis_results.get("top_genes", []),
        "pathogenic_variants": analysis_results.get("pathogenic_variants", []),
        "recommendations": analysis_results.get("recommendations", []),
        "methodology": "AI-based genomic analysis using ensemble models",
        "limitations": [
            "Results are for research/screening purposes",
            "Clinical interpretation required",
            "Not all genetic causes may be detected"
        ]
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    if format == "json":
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
    elif format == "html":
        html_content = _generate_html_report(report)
        with open(output_path, 'w') as f:
            f.write(html_content)
    
    return output_path


def _generate_html_report(report: Dict) -> str:
    """Generate HTML version of report."""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Hearing Screening Report - {report['sample_id']}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #333; }}
            .section {{ margin: 20px 0; }}
            .risk-high {{ color: red; font-weight: bold; }}
            .risk-moderate {{ color: orange; font-weight: bold; }}
            .risk-low {{ color: green; font-weight: bold; }}
        </style>
    </head>
    <body>
        <h1>Genomic Hearing Screening Report</h1>
        <p><strong>Report ID:</strong> {report['report_id']}</p>
        <p><strong>Sample ID:</strong> {report['sample_id']}</p>
        <p><strong>Generated:</strong> {report['generated_at']}</p>
        
        <div class="section">
            <h2>Risk Assessment</h2>
            <p>Risk Score: {report['risk_assessment']['overall_risk_score']:.2f}</p>
            <p>Category: <span class="risk-{report['risk_assessment']['risk_category']}">{report['risk_assessment']['risk_category'].upper()}</span></p>
        </div>
        
        <div class="section">
            <h2>Key Findings</h2>
            <ul>
                {''.join(f'<li>{f}</li>' for f in report['findings'])}
            </ul>
        </div>
        
        <div class="section">
            <h2>Recommendations</h2>
            <ul>
                {''.join(f'<li>{r}</li>' for r in report['recommendations'])}
            </ul>
        </div>
    </body>
    </html>
    """
    return html
