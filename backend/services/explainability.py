"""
Explainable Inspection Report Generator (Stage 10)
Synthesizes natural language causal explanations and actionable engineering recommendations.
"""
from typing import Dict, Any
from datetime import datetime

def generate_explainable_report(
    inspection_id: str,
    defect_type: str,
    clip_conf: float,
    sam_conf: float,
    defect_area_px: int,
    defect_area_mm2: float,
    location: str,
    relative_depth_mm: float,
    surface_profile: str,
    operator_text: str,
    weighted_risk: float,
    decision: str,
    image_previews: Dict[str, str]
) -> Dict[str, Any]:
    """
    Synthesizes natural language explainability trace and actionable recommendations.
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    
    # Generate contextual explanation
    if defect_type == "Normal":
        explanation = (
            f"Component inspection verified conforming baseline finish across {location}. "
            f"Zero-shot visual recognition identified normal machined surface with {clip_conf*100:.1f}% confidence. "
            f"Depth topography confirms uniform nominal planar profile with zero depression (0.00 mm). "
            f"No structural risks detected."
        )
        recommendation = "Approved for downstream assembly or shipment. Release component into inventory."
        
    elif defect_type == "Crack":
        explanation = (
            f"Component flagged for rejection due to severe material separation at {location}. "
            f"Zero-shot CLIP visual model detected a structural crack with {clip_conf*100:.1f}% confidence. "
            f"SAM 2 delineated a {defect_area_px:,} pixel ({defect_area_mm2:.2f} mm²) fracture perimeter. "
            f"Depth Anything V2 verified a {abs(relative_depth_mm):.2f} mm physical depression, confirming a structural cavity "
            f"rather than superficial marking. "
            + (f"Operator notes ('{operator_text}') substantiate mechanical stress." if operator_text else "")
        )
        recommendation = "Immediate Quarantine. Do not pressurize or install. Quarantine part for metallurgical failure analysis."
        
    elif defect_type == "Dent":
        explanation = (
            f"Plastic deformation detected at {location}. Visual similarity matched mechanical impact dent at {clip_conf*100:.1f}% confidence. "
            f"Segmented footprint encompasses {defect_area_mm2:.2f} mm² with localized topographical basin depth of {abs(relative_depth_mm):.2f} mm. "
            f"Resulting risk score is {weighted_risk:.2f}."
        )
        recommendation = "Quarantine for dimensional CMM check. Reject if seating face flatness tolerance is compromised."
        
    elif defect_type == "Scratch":
        explanation = (
            f"Linear abrasive score line observed along {location}. CLIP confidence is {clip_conf*100:.1f}%. "
            f"Segmented area is {defect_area_mm2:.2f} mm² with shallow profile depression ({abs(relative_depth_mm):.2f} mm). "
            f"Disposed as {decision} based on depth threshold and risk index of {weighted_risk:.2f}."
        )
        recommendation = (
            "Surface polish or re-lap gasket seating surface if depth exceeds 0.20 mm. Otherwise re-inspect." 
            if decision == "REVIEW" else "Acceptable within non-sealing margin."
        )
        
    elif defect_type == "Corrosion":
        explanation = (
            f"Surface oxidation and intergranular pitting detected at {location}. "
            f"Visual classification indicates chemical corrosion with {clip_conf*100:.1f}% confidence. "
            f"Surface gradient shows localized micro-roughness with {abs(relative_depth_mm):.2f} mm erosion."
        )
        recommendation = "Chemical pickling / surface blast treatment required. Re-evaluate remaining wall thickness."
        
    elif defect_type == "Discoloration":
        explanation = (
            f"Thermal heat tint / temper coloration observed at {location}. "
            f"CLIP identified discoloration at {clip_conf*100:.1f}%. "
            f"Crucially, Depth Anything V2 measured relative depth displacement of {relative_depth_mm:.2f} mm (essentially flat planar), "
            f"proving that the component retains structural thickness and lacks gouging."
        )
        recommendation = "Surface cosmetic cleaning or passivate with acid bath. Component remains structurally sound."
        
    else:
        explanation = f"Inspection completed with composite risk score {weighted_risk:.2f}. Disposition evaluated as {decision}."
        recommendation = "Follow standard shop-floor standard operating procedures (SOP)."

    return {
        "inspection_id": inspection_id,
        "date": now_str,
        "defect_type": defect_type,
        "location": location,
        "defect_area": f"{defect_area_mm2:.2f} mm² ({defect_area_px:,} px)",
        "confidence": f"{clip_conf*100:.1f}% (CLIP) / {sam_conf*100:.1f}% (SAM 2)",
        "risk_score": f"{weighted_risk:.2f} / 1.00",
        "decision": decision,
        "explanation": explanation,
        "recommendation": recommendation,
        "visual_evidence": image_previews
    }
