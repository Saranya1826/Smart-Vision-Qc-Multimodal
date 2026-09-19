"""
Pydantic Schemas for SmartVisionOC Inspection Pipeline
"""
from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field

class OperatorInputPayload(BaseModel):
    notes: Optional[str] = ""
    voice_transcript: Optional[str] = ""
    tool_wear_alert: bool = False
    critical_sealing_surface: bool = False
    quenching_anomaly: bool = False
    batch_recheck: bool = False

class InspectionConfig(BaseModel):
    enable_clahe: bool = True
    clahe_clip_limit: float = 2.0
    pass_threshold: float = 0.25
    reject_threshold: float = 0.65
    uncertainty_tolerance: float = 0.40

class InspectionRequest(BaseModel):
    specimen_id: Optional[str] = None
    image_base64: Optional[str] = None
    serial_number: Optional[str] = "SN-2026-FLG-9021"
    component_type: str = "ANSI B16.5 Class 300 Weld Neck Flange"
    operator_input: Optional[OperatorInputPayload] = None
    config: Optional[InspectionConfig] = None

# Stage Output Models
class ImageAcquisitionData(BaseModel):
    image_base64: str
    original_dimensions: Tuple[int, int]
    format: str
    file_size_kb: float
    sha256_hash: str
    capture_timestamp: str

class PreprocessingData(BaseModel):
    original_image_base64: str
    enhanced_image_base64: str
    quality_check: str
    focus_score: float = Field(..., description="Laplacian variance sharpness metric")
    brightness_score: float = Field(..., description="Mean normalized luminance [0, 100]")
    contrast_score: float = Field(..., description="RMS standard deviation contrast")
    noise_level_snr: float = Field(..., description="Estimated Signal-to-Noise Ratio (dB)")

class DefectIdentificationData(BaseModel):
    predicted_defect: str
    clip_confidence: float
    candidate_defect_types: List[str]
    similarity_scores: Dict[str, float]
    heatmap_base64: str
    peak_coordinate: Tuple[int, int]
    normal_probability: Optional[float] = 0.0
    defect_probability: Optional[float] = 0.0
    uncertainty: Optional[float] = 0.0
    anomaly_score: Optional[float] = 0.0
    classification_reason: Optional[str] = ""
    is_fallback: Optional[bool] = False
    model_status: Optional[str] = ""

class SegmentationData(BaseModel):
    mask_base64: str
    defect_area_px: int
    defect_area_mm2: float
    bounding_box: List[int] = Field(..., description="[x_min, y_min, x_max, y_max]")
    centroid: Tuple[int, int]
    sam_confidence: float
    polygon_points: List[List[int]]

class SpatialAnalysisData(BaseModel):
    depth_map_base64: str
    relative_depth_mm: float
    location: str
    surface_profile: str  # Cavity, Planar, Burr, Roughness
    surface_gradient: float

class OperatorInputData(BaseModel):
    text_input: str
    voice_input: str
    structured_observations: Dict[str, bool]
    parsed_severity_modifier: float

class EvidenceFusionData(BaseModel):
    visual_evidence: float
    operator_evidence: float
    evidence_agreement: float
    combined_confidence: float
    fusion_weights: Dict[str, float]
    fused_severity: float

class RiskScoringData(BaseModel):
    clip_confidence: float
    sam_confidence: float
    defect_area_score: float
    evidence_agreement: float
    uncertainty_penalty: float
    weighted_risk_score: float
    risk_level: str  # Nominal, Low, Moderate, High, Critical

class DecisionData(BaseModel):
    disposition: str  # PASS, REVIEW, REJECT
    status_label: str
    decision_reason: str
    safety_envelope_triggered: str

class ReportData(BaseModel):
    inspection_id: str
    date: str
    defect_type: str
    location: str
    defect_area: str
    confidence: str
    risk_score: str
    decision: str
    explanation: str
    recommendation: str
    visual_evidence: Dict[str, str]

class ProvenanceData(BaseModel):
    execution_mode: str
    disclaimer: str
    models_configured: Dict[str, str]
    inspection_type: str = "DEMO_PRESET"  # "DEMO_PRESET" | "USER_UPLOADED"

class InspectionResponse(BaseModel):
    inspection_id: str
    timestamp: str
    specimen_id: Optional[str] = None
    component_type: str
    provenance: ProvenanceData
    
    # 10 Stages
    section_1_acquisition: ImageAcquisitionData
    section_2_preprocessing: PreprocessingData
    section_3_defect_identification: DefectIdentificationData
    section_4_segmentation: SegmentationData
    section_5_spatial_analysis: SpatialAnalysisData
    section_6_operator_input: OperatorInputData
    section_7_evidence_fusion: EvidenceFusionData
    section_8_risk_scoring: RiskScoringData
    section_9_decision_engine: DecisionData
    section_10_inspection_report: ReportData

class SpecimenMeta(BaseModel):
    id: str
    name: str
    defect_type: str
    ground_truth_severity: str
    description: str
    thumbnail_base64: Optional[str] = None
