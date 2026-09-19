"""
Risk Engine, Decision Logic, and Causal Explainability Module
Computes transparent, calibrated risk scores in [0.0, 1.0], evaluates acceptance dispositions,
and synthesizes data-grounded causal inspection explanations and shop-floor directives.
"""
from typing import Dict, Any, Tuple
import numpy as np

DEFECT_SEVERITY_WEIGHTS = {
    "Normal": 0.0,
    "Contamination": 0.22,
    "Discoloration": 0.25,
    "Scratch": 0.38,
    "Dent": 0.52,
    "Corrosion": 0.68,
    "Hole": 0.85,
    "Missing Part": 0.88,
    "Deformation": 0.75,
    "Surface Damage": 0.82,
    "Other Anomaly": 0.40,
    "Crack": 0.92
}


def calculate_risk_and_decision(
    predicted_defect: str,
    clip_confidence: float,
    defect_area_pct: float,
    defect_area_px: int,
    sam_confidence: float,
    evidence_agreement: float,
    operator_evidence: float,
    critical_sealing_surface: bool = False,
    pass_threshold: float = 0.25,
    reject_threshold: float = 0.65,
    uncertainty_tolerance: float = 0.40
) -> Dict[str, Any]:
    """
    Computes calibrated weighted risk score [0.0, 1.0], risk level, and final disposition.
    """
    # 1. Defect Area Score
    area_score = float(np.clip(defect_area_pct / 4.0, 0.0, 1.0)) if predicted_defect != "Normal" else 0.0

    # 2. Uncertainty Penalty
    model_uncertainty = float(np.clip((1.0 - evidence_agreement) * 0.3 + (1.0 - clip_confidence) * 0.15, 0.0, 0.30))
    uncertainty_penalty = round(model_uncertainty if model_uncertainty > uncertainty_tolerance else 0.02, 2)

    # 3. Weighted Risk Computation
    if predicted_defect == "Normal" or defect_area_px == 0:
        base_risk = (1.0 - clip_confidence) * 0.10 + (operator_evidence * 0.08)
        weighted_risk = round(float(np.clip(base_risk, 0.02, pass_threshold - 0.08)), 2)
    else:
        sev_weight = DEFECT_SEVERITY_WEIGHTS.get(predicted_defect, 0.50)
        
        raw_risk = (
            (sev_weight * clip_confidence * 0.45) +
            (area_score * 0.30) +
            (operator_evidence * 0.15) +
            (uncertainty_penalty * 0.10)
        )
        # Critical surface safety rule
        if critical_sealing_surface and predicted_defect in ("Crack", "Dent", "Corrosion", "Hole", "Missing Part"):
            raw_risk = max(raw_risk, 0.72)

        weighted_risk = round(float(np.clip(raw_risk, 0.05, 0.99)), 2)

    # 4. Risk Level Categorization
    if weighted_risk <= 0.15:
        risk_level = "Nominal"
    elif weighted_risk <= pass_threshold:
        risk_level = "Low"
    elif weighted_risk < 0.50:
        risk_level = "Moderate"
    elif weighted_risk < reject_threshold:
        risk_level = "High"
    else:
        risk_level = "Critical"

    # 5. Disposition Evaluation
    safety_envelope = "NOMINAL_OPERATION"
    if critical_sealing_surface and predicted_defect in ("Crack", "Hole", "Missing Part") and defect_area_px > 0:
        disposition = "REJECT"
        status_label = "REJECT - CRITICAL SEALING SURFACE FRACTURE"
        safety_envelope = "CRITICAL_SURFACE_SAFETY_INTERLOCK"
    elif weighted_risk < pass_threshold:
        disposition = "PASS"
        status_label = "PASS - COMPONENT CONFORMING"
    elif weighted_risk >= reject_threshold:
        disposition = "REJECT"
        status_label = f"REJECT - CRITICAL {predicted_defect.upper()} DETECTED"
    else:
        disposition = "REVIEW"
        status_label = f"REVIEW - {predicted_defect.upper()} REQUIRES SECONDARY INSPECTION"

    # 6. Dynamic Causal Reason Generation
    if disposition == "PASS":
        reason = (
            f"No significant defect detected. Optical analysis indicates a nominal conforming component "
            f"(normal probability {round(clip_confidence * 100, 1)}%, calibrated risk {weighted_risk:.2f}). "
            f"Visual and segmentation metrics confirm conforming component profile."
        )
    elif disposition == "REJECT":
        reason = (
            f"High-confidence {predicted_defect.lower()} detected over {defect_area_pct:.2f}% "
            f"of the inspected surface ({round(clip_confidence * 100, 1)}% confidence, risk {weighted_risk:.2f}). "
            f"Cross-modal evidence agreement ({round(evidence_agreement * 100, 1)}%) corroborates out-of-tolerance defect."
        )
    else:
        reason = (
            f"Moderate {predicted_defect.lower()} detected with risk score {weighted_risk:.2f} "
            f"(confidence {round(clip_confidence * 100, 1)}%, footprint {defect_area_pct:.2f}%). "
            f"Manual secondary review mandated prior to release."
        )

    return {
        "weighted_risk_score": weighted_risk,
        "risk_level": risk_level,
        "disposition": disposition,
        "status_label": status_label,
        "decision_reason": reason,
        "safety_envelope_triggered": safety_envelope,
        "area_score": round(area_score, 2),
        "uncertainty_penalty": uncertainty_penalty
    }


def generate_explainable_narrative(
    inspection_id: str,
    predicted_defect: str,
    clip_confidence: float,
    defect_area_px: int,
    defect_area_pct: float,
    relative_depth_norm: float,
    surface_profile: str,
    location: str,
    weighted_risk: float,
    disposition: str,
    evidence_agreement: float,
    operator_evidence: float,
    is_fallback: bool
) -> Tuple[str, str]:
    """
    Synthesizes natural language causal narrative and actionable shop-floor recommendations.
    """
    method_note = "Inspection completed using fallback vision analysis." if is_fallback else "Inspection completed using PyTorch AI model suite."

    if disposition == "PASS":
        explanation = (
            f"No significant defect was detected in the inspected region ({location}). "
            f"Zero-shot normal confidence is {round(clip_confidence * 100, 1)}% with an estimated relative elevation "
            f"delta-z of {relative_depth_norm:+.2f} ({surface_profile}). "
            f"Evidence agreement between visual and metrological sensors is {round(evidence_agreement * 100, 1)}%. "
            f"Total calibrated risk is {weighted_risk:.2f}, within nominal acceptance bounds."
        )
        recommendation = f"Accept component. Cleared for production line assembly. {method_note}"
    elif disposition == "REJECT":
        explanation = (
            f"High-confidence {predicted_defect.lower()}-like evidence was detected in the localized region ({location}). "
            f"Visual classification confidence is {round(clip_confidence * 100, 1)}%, with a segmented defect footprint "
            f"of {defect_area_px:,} px ({defect_area_pct:.2f}% of inspected face). "
            f"Relative topographic profile indicates a localized depression of {relative_depth_norm:.2f} ({surface_profile}). "
            f"Evidence agreement ({round(evidence_agreement * 100, 1)}%) confirms structural discontinuity. "
            f"Total calibrated risk ({weighted_risk:.2f}) exceeds the critical reject threshold."
        )
        recommendation = f"Reject component and perform structural inspection. Quarantine lot immediately. {method_note}"
    else:
        explanation = (
            f"Surface anomaly evidence was detected and localized in the inspected component ({location}). "
            f"Visual confidence is {round(clip_confidence * 100, 1)}% covering {defect_area_px:,} px ({defect_area_pct:.2f}%). "
            f"Relative surface variation is {relative_depth_norm:.2f}. "
            f"Total calibrated risk is {weighted_risk:.2f}, triggering advisory hold."
        )
        recommendation = f"Review component manually. Require Level II Quality Inspector sign-off. {method_note}"

    return explanation, recommendation
