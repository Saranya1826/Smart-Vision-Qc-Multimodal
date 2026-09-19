"""
Evidence Fusion and Multi-Modal Agreement Engine
Fuses visual semantic evidence, metrology segmentation, 3D depth,
and human operator observations into a calibrated cross-modal confidence metric.
"""
from typing import Dict, Any, Tuple
import numpy as np


def parse_operator_observations(
    notes: str,
    voice_transcript: str,
    flags: Dict[str, bool]
) -> Tuple[float, float]:
    """
    Parses natural language notes and structured flags into an empirical severity score [0.0, 1.0].
    """
    combined_text = f"{notes} {voice_transcript}".lower().strip()
    
    critical_keywords = ["crack", "fracture", "rupture", "fail", "broken", "porosity", "leak"]
    moderate_keywords = ["chatter", "wear", "scratch", "dent", "tool wear", "rough", "gouge", "rust", "corrosion"]
    conforming_keywords = ["nominal", "pass", "ok", "pristine", "clean", "conforming", "normal"]

    text_score = 0.0
    if any(kw in combined_text for kw in critical_keywords):
        text_score = 0.85
    elif any(kw in combined_text for kw in moderate_keywords):
        text_score = 0.45
    elif any(kw in combined_text for kw in conforming_keywords):
        text_score = 0.05

    # Structured telemetry flags
    flag_score = 0.0
    if flags.get("critical_sealing_surface", False):
        flag_score += 0.25
    if flags.get("tool_wear_alert", False):
        flag_score += 0.20
    if flags.get("quenching_anomaly", False):
        flag_score += 0.20
    if flags.get("batch_recheck", False):
        flag_score += 0.15

    operator_evidence = float(np.clip(max(text_score, flag_score * 0.7) + (flag_score * 0.3), 0.0, 1.0))
    severity_bias = (operator_evidence - 0.2) * 0.15

    return round(operator_evidence, 3), round(severity_bias, 3)


def compute_evidence_fusion(
    predicted_defect: str,
    clip_confidence: float,
    defect_area_pct: float,
    defect_area_px: int,
    sam_confidence: float,
    relative_depth_norm: float,
    operator_evidence: float,
    fusion_weights: Dict[str, float] = None
) -> Dict[str, Any]:
    """
    Calculates multi-modal evidence fusion, evidence agreement, and combined confidence.
    Prevents false positive amplification if segmentation finds zero localized region.
    """
    weights = fusion_weights or {
        "visual": 0.35,
        "geometric": 0.25,
        "depth": 0.25,
        "operator": 0.15
    }

    # 1. Visual Evidence [0.0, 1.0]
    if predicted_defect == "Normal":
        visual_evidence = round(max(0.01, float(1.0 - clip_confidence)), 3)
    else:
        visual_evidence = round(float(clip_confidence), 3)

    # 2. Geometric / Segmentation Evidence [0.0, 1.0]
    if predicted_defect == "Normal" or defect_area_px == 0:
        area_score = 0.0
        geometric_evidence = 0.0
    else:
        area_score = float(np.clip(defect_area_pct / 4.0, 0.0, 1.0))
        geometric_evidence = round(float((area_score * 0.65) + (sam_confidence * 0.35)), 3)

    # 3. Depth Evidence [0.0, 1.0]
    depth_score = float(np.clip(abs(relative_depth_norm), 0.0, 1.0)) if predicted_defect != "Normal" else 0.0
    depth_evidence = round(depth_score, 3)

    # 4. Cross-Modal Evidence Agreement [0.0, 1.0]
    if predicted_defect == "Normal":
        # Conforming part: high agreement when all modalities indicate nominal
        agreement = max(0.85, round(float(clip_confidence * 0.95), 3))
    else:
        # Check consistency: if visual says defect but segmentation finds 0 px, agreement collapses
        if defect_area_px == 0:
            agreement = 0.25  # High contradiction
        else:
            # High agreement when visual and segmentation both corroborate
            signals = [visual_evidence, geometric_evidence, depth_evidence]
            var = float(np.var(signals))
            agreement = round(float(np.clip(1.0 - (var * 2.2), 0.40, 0.98)), 3)

    evidence_agreement = round(agreement, 3)

    # 5. Combined Confidence
    if predicted_defect == "Normal":
        combined_confidence = round(float(clip_confidence * (0.85 + 0.15 * evidence_agreement)), 3)
    else:
        # If contradiction exists (zero defect pixels), pull back confidence
        if defect_area_px == 0:
            combined_confidence = round(float(visual_evidence * 0.4), 3)
        else:
            fused_sig = (
                weights["visual"] * visual_evidence +
                weights["geometric"] * geometric_evidence +
                weights["depth"] * depth_evidence +
                weights["operator"] * operator_evidence
            )
            combined_confidence = round(float(np.clip(fused_sig * evidence_agreement, 0.10, 0.99)), 3)

    uncertainty = round(float(np.clip(1.0 - combined_confidence, 0.02, 0.85)), 3)

    fused_severity = round(float(
        weights["visual"] * visual_evidence +
        weights["geometric"] * geometric_evidence +
        weights["depth"] * depth_evidence +
        weights["operator"] * operator_evidence
    ), 3)

    return {
        "visual_evidence": visual_evidence,
        "operator_evidence": operator_evidence,
        "evidence_agreement": evidence_agreement,
        "combined_confidence": combined_confidence,
        "uncertainty": uncertainty,
        "fusion_weights": weights,
        "fused_severity": fused_severity,
        "area_score": round(area_score, 3)
    }
