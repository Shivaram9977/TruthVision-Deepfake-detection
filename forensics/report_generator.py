import os
from typing import Dict, Any
from .risk_scorer import ForensicAssessment


def generate_html_report(assessment: ForensicAssessment, output_path: str = None) -> str:
    """
    Renders a forensic evaluation report as a standalone printable Black & White HTML document.
    """
    badge_bg = "#ffffff" if assessment.classification == "REAL" else "#000000"
    badge_fg = "#000000" if assessment.classification == "REAL" else "#ffffff"
    badge_border = "#ffffff"

    suspicious_tags = "".join([f'<span class="badge-tag">{zone}</span>' for zone in assessment.suspicious_regions])
    interval_tags = "".join([f'<span class="badge-tag interval-tag">{interval}</span>' for interval in assessment.flagged_intervals])

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>TruthVision Forensic Report - {assessment.media_name}</title>
    <style>
        body {{
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
            background-color: #000000;
            color: #ffffff;
            margin: 0;
            padding: 40px;
        }}
        .report-card {{
            max-width: 850px;
            margin: 0 auto;
            background-color: #0d0d0d;
            border-radius: 12px;
            padding: 40px;
            border: 1px solid #333333;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.9);
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid #262626;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .logo-title {{
            font-size: 24px;
            font-weight: 800;
            color: #ffffff;
            letter-spacing: -0.5px;
            text-transform: uppercase;
        }}
        .report-badge {{
            background-color: {badge_bg};
            color: {badge_fg};
            border: 1px solid {badge_border};
            padding: 6px 18px;
            border-radius: 20px;
            font-weight: 800;
            font-size: 15px;
            letter-spacing: 0.5px;
        }}
        .grid-2 {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 25px;
        }}
        .metric-box {{
            background-color: #000000;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #262626;
        }}
        .metric-label {{
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #737373;
            margin-bottom: 6px;
            font-weight: 600;
        }}
        .metric-value {{
            font-size: 22px;
            font-weight: 700;
            color: #ffffff;
        }}
        .progress-bar-bg {{
            background-color: #262626;
            height: 6px;
            border-radius: 3px;
            margin-top: 10px;
            overflow: hidden;
        }}
        .progress-bar-fill {{
            height: 100%;
            background-color: #ffffff;
            width: {assessment.risk_score}%;
        }}
        .badge-tag {{
            display: inline-block;
            background-color: #171717;
            color: #ffffff;
            border: 1px solid #333333;
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 12px;
            margin: 3px 3px 3px 0;
            font-family: monospace;
        }}
        .interval-tag {{
            background-color: #000000;
            border-color: #ffffff;
            color: #ffffff;
            font-weight: bold;
        }}
        .summary-text {{
            line-height: 1.6;
            color: #a3a3a3;
            font-size: 14px;
            background-color: #000000;
            padding: 20px;
            border-radius: 8px;
            border-left: 3px solid #ffffff;
            border-top: 1px solid #262626;
            border-right: 1px solid #262626;
            border-bottom: 1px solid #262626;
        }}
        .footer {{
            margin-top: 35px;
            text-align: center;
            font-size: 11px;
            color: #525252;
            border-top: 1px solid #262626;
            padding-top: 20px;
        }}
        @media print {{
            body {{ background-color: #ffffff; color: #000000; padding: 0; }}
            .report-card {{ border: 1px solid #000000; box-shadow: none; background: #ffffff; color: #000000; }}
            .metric-box {{ background-color: #ffffff; border: 1px solid #000000; color: #000000; }}
            .metric-value {{ color: #000000; }}
            .summary-text {{ background-color: #ffffff; color: #000000; border-color: #000000; }}
            .logo-title {{ color: #000000; }}
            .report-badge {{ background-color: #000000; color: #ffffff; }}
        }}
    </style>
</head>
<body>
    <div class="report-card">
        <div class="header">
            <div>
                <div class="logo-title">TruthVision Forensic Engine</div>
                <div style="color: #737373; font-size: 12px; margin-top: 4px;">Automated Deepfake Authenticity Certificate</div>
            </div>
            <div class="report-badge">{assessment.classification}</div>
        </div>

        <div class="grid-2">
            <div class="metric-box">
                <div class="metric-label">Target Media Name</div>
                <div class="metric-value" style="font-size: 16px; word-break: break-all;">{assessment.media_name}</div>
                <div style="font-size: 12px; color: #737373; margin-top: 6px;">Type: {assessment.media_type} | Timestamp: {assessment.timestamp}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Composite Risk Index</div>
                <div class="metric-value">{assessment.risk_score:.1f}% ({assessment.risk_level})</div>
                <div class="progress-bar-bg">
                    <div class="progress-bar-fill"></div>
                </div>
            </div>
        </div>

        <div class="grid-2">
            <div class="metric-box">
                <div class="metric-label">Neural Synthetic Likelihood</div>
                <div class="metric-value">{assessment.deepfake_probability:.1f}%</div>
                <div style="font-size: 11px; color: #737373; margin-top: 4px;">Calculated via EfficientNet-B0 transfer backbone</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Authenticity Confidence</div>
                <div class="metric-value">{assessment.authenticity_score:.1f}%</div>
                <div style="font-size: 11px; color: #737373; margin-top: 4px;">Inverse weighted consistency measure</div>
            </div>
        </div>

        <div class="metric-box" style="margin-bottom: 25px;">
            <div class="metric-label">Likely Manipulation Typology</div>
            <div style="font-size: 16px; font-weight: 600; color: #ffffff; margin: 4px 0 12px 0;">{assessment.likely_technique}</div>
            <div class="metric-label" style="margin-top: 10px;">Identified Saliency Zones:</div>
            <div>{suspicious_tags}</div>
            {"<div class='metric-label' style='margin-top: 12px;'>Suspicious Time Intervals:</div><div>" + interval_tags + "</div>" if assessment.media_type == "VIDEO" else ""}
        </div>

        <div class="summary-text">
            <strong style="color: #ffffff;">Forensic Assessment Summary:</strong><br>
            {assessment.summary_notes}
        </div>

        <div class="footer">
            Generated by TruthVision Forensic Intelligence Platform &bull; PyTorch Neural Architecture &bull; ISO/IEC Aligned Synthetic Media Verification
        </div>
    </div>
</body>
</html>"""

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

    return html_content
