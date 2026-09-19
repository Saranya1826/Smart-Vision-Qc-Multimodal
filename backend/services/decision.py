"""
Deterministic Decision Engine with Safety Bounds (Stage 9)
Evaluates calibrated risk against configurable pass/reject thresholds.
"""
from typing import Dict, Any

def evaluate_inspection_decision(
    weighted_risk: float,
    uncertainty_penalty: float,
    predicted_defect: str,
    pass_threshold: float = 0.25,
    reject_threshold: float = 0.65,
    uncertainty_tolerance: float = 0.40
) -> Dict[str, Any]:
    """
    Renders deterministic PASS, REVIEW, or REJECT disposition with safety hysteresis.
    """
    if predicted_defect == "Normal" and weighted_risk < pass_threshold:
        return {
            "disposition": "PASS",
            "status_label": "CONFORMING COMPONENT - PASS",
            "decision_reason": "Zero structural anomalies identified. Surface geometry and phonographic finish meet ISO/ASME specifications.",
            "safety_envelope_triggered": "NOMINAL_ACCEPTANCE"
        }
        
    if weighted_risk >= reject_threshold:
        return {
            "disposition": "REJECT",
            "status_label": "NON-CONFORMING DEFECT - REJECT",
            "decision_reason": f"High risk score ({weighted_risk:.2f}) exceeds rejection threshold ({reject_threshold:.2f}). Critical {predicted_defect} poses integrity failure hazard.",
            "safety_envelope_triggered": "CRITICAL_DEFECT_CONTAINMENT"
        }
        
    if uncertainty_penalty >= uncertainty_tolerance:
        return {
            "disposition": "REVIEW",
            "status_label": "OPERATOR VERIFICATION REQUIRED - REVIEW",
            "decision_reason": f"Model uncertainty ({uncertainty_penalty:.2f}) exceeds tolerance ({uncertainty_tolerance:.2f}). Requires manual metrology verification.",
            "safety_envelope_triggered": "HIGH_UNCERTAINTY_FAILSAFE"
        }
        
    if weighted_risk < pass_threshold:
        return {
            "disposition": "PASS",
            "status_label": "CONFORMING WITH MINOR VARIATION - PASS",
            "decision_reason": f"Risk score ({weighted_risk:.2f}) below threshold ({pass_threshold:.2f}). Anomaly is cosmetic or within allowable engineering tolerances.",
            "safety_envelope_triggered": "TOLERANCE_MARGIN_PASS"
        }
    else:
        return {
            "disposition": "REVIEW",
            "status_label": "BORDERLINE CONDITION - REVIEW",
            "decision_reason": f"Risk score ({weighted_risk:.2f}) falls in ambiguous margin [{pass_threshold:.2f} - {reject_threshold:.2f}]. Secondary QA sign-off mandatory.",
            "safety_envelope_triggered": "BORDERLINE_HYSTERESIS_HOLD"
        }
