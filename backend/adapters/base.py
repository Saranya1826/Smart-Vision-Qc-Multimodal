"""
Base Abstract Classes for Pluggable AI Adapters
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
import numpy as np

class BaseZeroShotDetector(ABC):
    """Abstract interface for Zero-Shot Open-Vocabulary Defect Identification (CLIP)."""
    
    @abstractmethod
    def identify_defects(
        self,
        image_rgb: np.ndarray,
        candidate_classes: List[str],
        prompts: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """
        Identify defect probabilities and generate coarse visual activation heatmap.
        """
        pass

class BaseSegmenter(ABC):
    """Abstract interface for Promptable Defect Segmentation (SAM 2)."""
    
    @abstractmethod
    def segment_defect(
        self,
        image_rgb: np.ndarray,
        point_prompts: List[Tuple[int, int]],
        box_prompt: List[int]
    ) -> Dict[str, Any]:
        """
        Generate pixel-level binary mask and vector polygon contour for defect.
        """
        pass

class BaseDepthEstimator(ABC):
    """Abstract interface for Monocular Depth / Topographical Analysis (Depth Anything V2)."""
    
    @abstractmethod
    def estimate_depth(
        self,
        image_rgb: np.ndarray,
        defect_mask: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Generate relative depth map and quantify surface relief / depression delta z.
        """
        pass
