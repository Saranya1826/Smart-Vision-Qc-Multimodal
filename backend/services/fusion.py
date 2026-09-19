"""
Evidence Fusion and Calibrated Risk Scoring Service (Stages 6, 7, 8)
Synthesizes visual, geometric, depth, and operator evidence streams.
"""
from typing import Dict, Any, Tuple
import numpy as np

# Base defect intrinsic hazard weights
INTRINSIC_DEFECT_SEVERITY = {
    "Crack": 1.00,        # Critical structural failure hazard
    "Dent": 0.80,         # High plastic deformation
    "Corrosion": 0.75,    # High oxidation/pitting
    "Scratch": 0.45,      # Moderate superficial score
    "Discoloration": 0.20,# Low cosmetic/thermal patina
    "Normal": 0.00        # Zero defect
}

def parse_operator_input(
    notes: str,
    voice_transcript: str,
    flags: Dict[str, bool]
) -> Tuple[float, float]:
    """
    Parses operator textual context and checkboxes into an operator severity index [0, 1]
    and bias modifier [-0.3, +0.3].
    """
    combined_text = (notes + " " + voice_transcript).lower()
    
    # Base prior from checkboxes
    flag_score = 0.0
    if flags.get("critical_sealing_surface", False):
        flag_score += 0.35
    if flags.get("tool_wear_alert", False):
        flag_score += 0.25
    if flags.get("quenching_anomaly", False):
        flag_score += 0.20
    if flags.get("batch_recheck", False):
        flag_score += 0.15
        
    # Textual keyword sentiment modifier
    text_modifier = 0.0
    aggravating_keywords = ["crack", "fracture", "deep", "vibration", "damage", "severe", "gouge", "reject", "fail"]
    mitigating_keywords = ["cosmetic", "superficial", "stain", "nominal", "wash", "water", "light", "pass"]
    
    for kw in aggravating_keywords:
        if kw in combined_text:
            text_modifier += 0.12
    for kw in mitigating_keywords:
        if kw in combined_text:
            text_modifier -= 0.10
            
    operator_severity = max(0.0, min(1.0, flag_score + max(-0.2, min(0.4, text_modifier))))
    bias_modifier = max(-0.25, min(0.25, (operator_severity - 0.3) * 0.5))
    
    return round(operator_severity, 3), round(bias_modifier, 3)

def compute_evidence_fusion(
    predicted_defect: str,
    clip_conf: float,
    sam_conf: float,
    defect_area_px: int,
    relative_depth_mm: float,
    operator_severity: float,
    custom_weights: Dict[str, float] = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Computes Section 7 (Evidence Fusion) and Section 8 (Risk Scoring).
    """
    weights = custom_weights or {
        "visual": 0.35,
        "geometric": 0.20,
        "depth": 0.30,
        "operator": 0.15
    }
    
    # 1. Visual Severity
    class_hazard = INTRINSIC_DEFECT_SEVERITY.get(predicted_defect, 0.5)
    visual_evidence = clip_conf * class_hazard
    
    # 2. Geometric Severity (Area & SAM confidence)
    # Area normalization: 5000px is considered critical threshold on 1024x1024
    area_norm = min(1.0, defect_area_px / 4500.0)
    geometric_evidence = area_norm * sam_conf
    
    # 3. Depth Severity (Topological depression)
    if relative_depth_mm < -0.10: # True depression
        depth_severity = min(1.0, abs(relative_depth_mm) / 1.8)
    else:
        # Discoloration or flat surface
        depth_severity = 0.05
        
    # 4. Evidence Agreement
    # Check alignment between visual severity and operator severity
    visual_flag = visual_evidence > 0.4
    operator_flag = operator_severity > 0.3
    if visual_flag == operator_flag:
        evidence_agreement = round(1.0 - abs(visual_evidence - operator_severity) * 0.4, 3)
    else:
        evidence_agreement = round(max(0.2, 1.0 - abs(visual_evidence - operator_severity)), 3)
        
    # 5. Combined Confidence
    combined_confidence = round(0.55 * clip_conf + 0.35 * sam_conf + 0.10 * evidence_agreement, 3)
    
    # 6. Weighted Fused Score
    fused_severity = (
        weights["visual"] * visual_evidence +
        weights["geometric"] * geometric_evidence +
        weights["depth"] * depth_severity +
        weights["operator"] * operator_severity
    )
    
    # 7. Uncertainty Estimation
    # Epistemic disagreement between visual defect claim and flat depth
    if class_hazard > 0.7 and depth_severity < 0.15:
        epistemic_penalty = 0.25 # Model contradiction: visual says crack, but surface is completely flat
    else:
        epistemic_penalty = 0.03
        
    uncertainty_penalty = round(epistemic_penalty + (1.0 - combined_confidence) * 0.18, 3)
    
    # 8. Calibrated Risk Score [0.00, 1.00]
    if predicted_defect == "Normal":
        weighted_risk = round(max(0.01, 0.04 + operator_severity * 0.1), 3)
    else:
        weighted_risk = round(min(1.0, fused_severity + uncertainty_penalty * 0.4), 3)
        
    # Risk Level Category
    if weighted_risk >= 0.75:
        risk_level = "Critical"
    elif weighted_risk >= 0.55:
        risk_level = "High"
    elif weighted_risk >= 0.30:
        risk_level = "Moderate"
    elif weighted_risk >= 0.12:
        risk_level = "Low"
    else:
        risk_level = "Nominal"
        
    fusion_data = {
        "visual_evidence": round(visual_evidence, 3),
        "operator_evidence": round(operator_severity, 3),
        "evidence_agreement": evidence_agreement,
        "combined_confidence": combined_confidence,
        "fusion_weights": weights,
        "fused_severity": round(fused_severity, 3)
    }
    
    risk_data = {
        "clip_confidence": round(clip_conf, 3),
        "sam_confidence": round(sam_conf, 3),
        "defect_area_score": round(geometric_evidence, 3),
        "evidence_agreement": evidence_agreement,
        "uncertainty_penalty": uncertainty_penalty,
        "weighted_risk_score": weighted_risk,
        "risk_level": risk_level
    }
    
    return fusion_data, risk_data
