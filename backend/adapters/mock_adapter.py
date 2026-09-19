"""
Mock AI Adapters for CLIP, SAM 2, and Depth Anything V2
Implements realistic deterministic simulation using synthetic defect parameters.
"""
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from backend.adapters.base import BaseZeroShotDetector, BaseSegmenter, BaseDepthEstimator
from backend.services.mock_data import (
    generate_mock_heatmap,
    generate_mock_sam_mask,
    generate_mock_depth_map,
    SPECIMEN_CATALOG
)

DEFECT_TAXONOMY = ["Crack", "Scratch", "Dent", "Corrosion", "Discoloration", "Normal"]

class MockZeroShotDetector(BaseZeroShotDetector):
    """
    High-fidelity deterministic mock adapter for Zero-Shot CLIP defect detection.
    Generates class probability distributions and 2D visual attention heatmaps.
    """
    def __init__(self, default_specimen: str = "FLG-CRK-01"):
        self.default_specimen = default_specimen
        
    def identify_defects(
        self,
        image_rgb: np.ndarray,
        candidate_classes: List[str] = DEFECT_TAXONOMY,
        prompts: Optional[Dict[str, List[str]]] = None,
        specimen_id: Optional[str] = None
    ) -> Dict[str, Any]:
        spec_id = specimen_id or self.default_specimen
        spec = SPECIMEN_CATALOG.get(spec_id, SPECIMEN_CATALOG["FLG-CRK-01"])
        params = spec["defect_params"]
        primary = spec["defect_type"]
        base_conf = params["base_conf"]
        
        # Build normalized probability distribution
        scores = {}
        remaining_prob = 1.0 - base_conf
        other_classes = [c for c in candidate_classes if c != primary]
        
        scores[primary] = round(base_conf, 3)
        # Distribute remaining probability realistically
        weights = np.random.dirichlet(np.ones(len(other_classes)))
        for c, w in zip(other_classes, weights):
            scores[c] = round(float(w * remaining_prob), 3)
            
        # Re-normalize to sum to 1.0
        total = sum(scores.values())
        scores = {k: round(v / total, 3) for k, v in scores.items()}
        
        # Generate 2D Heatmap
        heatmap_arr, heatmap_b64 = generate_mock_heatmap(image_rgb.shape, params)
        
        return {
            "predicted_defect": primary,
            "clip_confidence": scores[primary],
            "candidate_defect_types": candidate_classes,
            "similarity_scores": scores,
            "heatmap_base64": heatmap_b64,
            "peak_coordinate": params["center"],
            "inference_mode": "MOCK_SIMULATION"
        }

class MockSegmenter(BaseSegmenter):
    """
    High-fidelity deterministic mock adapter for SAM 2 promptable segmentation.
    Produces pixel-level binary masks, polygon coordinates, and area measurements.
    """
    def __init__(self, default_specimen: str = "FLG-CRK-01"):
        self.default_specimen = default_specimen
        
    def segment_defect(
        self,
        image_rgb: np.ndarray,
        point_prompts: Optional[List[Tuple[int, int]]] = None,
        box_prompt: Optional[List[int]] = None,
        specimen_id: Optional[str] = None
    ) -> Dict[str, Any]:
        spec_id = specimen_id or self.default_specimen
        spec = SPECIMEN_CATALOG.get(spec_id, SPECIMEN_CATALOG["FLG-CRK-01"])
        params = spec["defect_params"]
        cx, cy = params["center"]
        sw, sh = params["size"]
        
        mask_arr, polygon_pts, mask_b64, area_px = generate_mock_sam_mask(image_rgb.shape, params)
        
        # Compute bounding box
        if params["type"] == "normal":
            bbox = [0, 0, 0, 0]
            centroid = (0, 0)
            sam_conf = 0.985
            area_mm2 = 0.0
        else:
            bbox = [
                max(0, cx - sw // 2 - 5),
                max(0, cy - sh // 2 - 5),
                min(image_rgb.shape[1], cx + sw // 2 + 5),
                min(image_rgb.shape[0], cy + sh // 2 + 5)
            ]
            centroid = (cx, cy)
            sam_conf = round(float(0.91 + np.random.uniform(0.01, 0.06)), 3)
            # Physical conversion factor approx: 1 pixel ~ 0.0044 mm^2 at 1024x1024 on 300mm flange
            area_mm2 = round(area_px * 0.0044, 2)
            
        return {
            "mask_base64": mask_b64,
            "defect_area_px": area_px,
            "defect_area_mm2": area_mm2,
            "bounding_box": bbox,
            "centroid": centroid,
            "sam_confidence": sam_conf,
            "polygon_points": polygon_pts,
            "inference_mode": "MOCK_SIMULATION"
        }

class MockDepthEstimator(BaseDepthEstimator):
    """
    High-fidelity deterministic mock adapter for Depth Anything V2 monocular depth estimation.
    Computes relative surface elevation, depression delta z, and 3D surface profile.
    """
    def __init__(self, default_specimen: str = "FLG-CRK-01"):
        self.default_specimen = default_specimen
        
    def estimate_depth(
        self,
        image_rgb: np.ndarray,
        defect_mask: Optional[np.ndarray] = None,
        specimen_id: Optional[str] = None
    ) -> Dict[str, Any]:
        spec_id = specimen_id or self.default_specimen
        spec = SPECIMEN_CATALOG.get(spec_id, SPECIMEN_CATALOG["FLG-CRK-01"])
        params = spec["defect_params"]
        
        depth_arr, depth_b64 = generate_mock_depth_map(image_rgb.shape, params)
        
        delta_z = params["delta_z_mm"]
        profile = params["profile"]
        location = params["location_name"]
        gradient = 0.78 if abs(delta_z) > 1.0 else (0.42 if abs(delta_z) > 0.2 else 0.08)
        
        return {
            "depth_map_base64": depth_b64,
            "relative_depth_mm": delta_z,
            "location": location,
            "surface_profile": profile,
            "surface_gradient": gradient,
            "inference_mode": "MOCK_SIMULATION"
        }
