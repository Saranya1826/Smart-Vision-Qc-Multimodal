# Engineering Architecture & Implementation Specification
## SmartVisionOC: Explainable Multi-Modal AI Framework for Zero-Shot Industrial Quality Inspection

---

### Document Information
- **Document Version**: 1.0.0
- **Document Type**: Technical Architecture & Engineering Specification
- **Target System**: Full-Stack Industrial Web Application & Modular AI Service
- **Execution Target**: Node.js 18+ / Vite / React 18 frontend + Python 3.10+ FastAPI backend (with zero-dependency fallback / mock execution capability)

---

## 1. System Architecture Overview

SmartVisionOC is engineered as a **decoupled, event-driven multi-modal inspection platform**. It separates client-side visual telemetry rendering from backend algorithmic execution via a strictly typed RESTful API and pluggable AI adapters.

```
+---------------------------------------------------------------------------------------------------------+
|                                          FRONTEND ARCHITECTURE                                          |
|                                       (React 18 + Vite + Tailwind CSS)                                  |
|                                                                                                         |
|  +---------------------------+  +--------------------------------+  +--------------------------------+  |
|  |     Control Sidebar       |  |     Multi-Layer Canvas         |  |    Telemetry & Decision Card   |  |
|  | - Upload / Preset Loader  |  | - WebGL / Canvas2D Renderer    |  | - Dynamic Gauge (0.0 to 1.0)   |  |
|  | - Preprocessing Sliders   |  | - Alpha-blended Heatmap Layer  |  | - Softmax Defect Spectrum      |  |
|  | - Operator Notes & Flags  |  | - Vector Polygon Mask Layer    |  | - Depth Topography Metrics     |  |
|  | - Manual Prompt Injector  |  | - Pseudo-color Depth Overlay   |  | - Explainability Text Engine   |  |
|  +---------------------------+  +--------------------------------+  +--------------------------------+  |
+---------------------------------------------------|-----------------------------------------------------+
                                                    | HTTP REST / WebSockets
                                                    v
+---------------------------------------------------------------------------------------------------------+
|                                      BACKEND SERVICE ARCHITECTURE                                       |
|                                        (FastAPI / Python 3.10+)                                         |
|                                                                                                         |
|  +---------------------------------------------------------------------------------------------------+  |
|  | Pipeline Orchestrator (10-Stage Sequential & Parallel Graph)                                      |  |
|  +---------------------------------------------------------------------------------------------------+  |
|         |                     |                      |                    |                  |          |
|         v                     v                      v                    v                  v          |
|  +---------------+  +-------------------+  +-------------------+  +---------------+  +---------------+  |
|  | Image Preproc |  | Zero-Shot CLIP    |  | SAM 2 Segmentation|  | Depth Anything|  | Evidence      |  |
|  | Service       |  | Adapter (Pluggable|  | Adapter (Pluggable|  | V2 Adapter     |  | Fusion & Risk |  |
|  | - CLAHE       |  | - Mock Engine     |  | - Mock Engine     |  | - Mock Engine |  | Engine        |  |
|  | - Bilateral   |  | - OpenCLIP PyTorch|  | - PyTorch SAM 2   |  | - Depth PyTorch|  | - Tri-State   |  |
|  | - Normalizer  |  | - TensorRT / ONNX |  | - ONNX Runtime    |  | - ONNX Runtime|  |   Decision    |  |
|  +---------------+  +-------------------+  +-------------------+  +---------------+  +---------------+  |
|                                                                                              |          |
|                                                                                              v          |
|                                                                                      +---------------+  |
|                                                                                      | Explainability|  |
|                                                                                      | & Report Gen  |  |
|                                                                                      | - Rationale   |  |
|                                                                                      | - PDF / JSON  |  |
|                                                                                      +---------------+  |
+---------------------------------------------------------------------------------------------------------+
```

---

## 2. Pluggable AI Adapter Pattern (Zero-Vendor-Lock)

To satisfy the strict constraint of **initial mock execution** while ensuring **seamless production upgradability** to live PyTorch or ONNX backends, the system employs the **Abstract Strategy / Adapter Pattern**.

### 2.1 Base Class Interfaces

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
import numpy as np

class BaseZeroShotDetector(ABC):
    """Abstract Base Class for CLIP-style zero-shot defect identification."""
    
    @abstractmethod
    def identify_defects(
        self, 
        image: np.ndarray, 
        candidate_classes: List[str], 
        prompt_templates: List[str]
    ) -> Dict[str, Any]:
        """
        Returns:
            {
                "class_probabilities": {"crack": 0.89, "scratch": 0.05, ...},
                "primary_defect": "crack",
                "confidence": 0.89,
                "attention_heatmap": np.ndarray (H, W float in [0, 1]),
                "peak_coordinate": (x, y),
                "bounding_box": [x_min, y_min, x_max, y_max],
                "inference_mode": "MOCK" | "PYTORCH_CUDA" | "ONNX"
            }
        """
        pass

class BaseSegmenter(ABC):
    """Abstract Base Class for SAM 2 promptable segmentation."""
    
    @abstractmethod
    def segment_defect(
        self, 
        image: np.ndarray, 
        point_prompts: List[Tuple[int, int]], 
        box_prompt: List[int]
    ) -> Dict[str, Any]:
        """
        Returns:
            {
                "binary_mask": np.ndarray (H, W bool),
                "polygon_contours": List[List[Tuple[int, int]]],
                "area_pixels": int,
                "aspect_ratio": float,
                "iou_score": float,
                "inference_mode": "MOCK" | "PYTORCH_CUDA" | "ONNX"
            }
        """
        pass

class BaseDepthEstimator(ABC):
    """Abstract Base Class for Depth Anything V2 monocular topological estimation."""
    
    @abstractmethod
    def estimate_depth(
        self, 
        image: np.ndarray, 
        defect_mask: np.ndarray
    ) -> Dict[str, Any]:
        """
        Returns:
            {
                "relative_depth_map": np.ndarray (H, W float in [0, 1]),
                "defect_depth_delta": float (e.g. -1.82 mm relative depression),
                "surface_roughness_gradient": float,
                "topography_type": "DEEP_CAVITY" | "SURFACE_PLANAR" | "BURR_PROTRUSION",
                "inference_mode": "MOCK" | "PYTORCH_CUDA" | "ONNX"
            }
        """
        pass
```

### 2.2 Factory & Dependency Injection
A central `ModelRegistry` instantiates the appropriate adapter based on environment variables:
```python
# Environment configuration
AI_INFERENCE_BACKEND = os.getenv("AI_INFERENCE_BACKEND", "MOCK")  # "MOCK" | "TORCH" | "ONNX"

def get_zero_shot_detector() -> BaseZeroShotDetector:
    if AI_INFERENCE_BACKEND == "TORCH":
        from backend.services.torch_clip_adapter import TorchCLIPDetector
        return TorchCLIPDetector()
    elif AI_INFERENCE_BACKEND == "ONNX":
        from backend.services.onnx_clip_adapter import OnnxCLIPDetector
        return OnnxCLIPDetector()
    else:
        from backend.services.mock_ai_adapter import MockZeroShotDetector
        return MockZeroShotDetector()
```

### 2.3 Transparent AI Reality Guarantee
Every JSON response payload returned by the inspection API includes an immutable provenance block:
```json
{
  "provenance": {
    "execution_mode": "MOCK_SIMULATION",
    "disclaimer": "SIMULATION ONLY: Results generated by deterministic algorithmic mock engine for testing and architectural validation. No neural network inference performed.",
    "model_versions": {
      "clip_backbone": "MOCK-ViT-L/14@336px",
      "sam2_backbone": "MOCK-SAM-2-Hiera-Large",
      "depth_backbone": "MOCK-Depth-Anything-V2-Large"
    },
    "gpu_accelerated": false
  }
}
```

---

## 3. Algorithmic Specifications & Mathematical Formulations

### 3.1 Preprocessing Pipeline (Stage 2)
1. **Contrast Limited Adaptive Histogram Equalization (CLAHE)**:
   - Divides the image into contextual tiles of size $8 \times 8$ pixels.
   - Clipping limit $\beta = 2.0$ prevents noise amplification in uniform alloy zones.
   - Bilinear interpolation between neighboring tiles to eliminate artificial boundary seams.
2. **Bilateral Filtering**:
   $$I_{filtered}(x) = \frac{1}{W_p} \sum_{x_i \in \Omega} I(x_i) f_r(\|I(x_i) - I(x)\|) g_s(\|x_i - x\|)$$
   Smooths metallic crystalline grain without softening high-contrast crack edges.

### 3.2 Evidence Fusion Engine (Stage 7)
The framework employs a **Multi-Criteria Bayesian Fusion Scheme** to combine disparate modalities:
1. **Visual Evidence ($S_{vis}$)**:
   $$S_{vis} = P(C^*) \cdot \beta_{class}(C^*)$$
   Where $P(C^*)$ is the top zero-shot class probability, and $\beta_{class}$ is the intrinsic defect severity weight (e.g., Crack $= 1.0$, Dent $= 0.8$, Discoloration $= 0.25$).
2. **Geometric Severity ($S_{geo}$)**:
   $$S_{geo} = \min\left(1.0, \frac{\text{Area}_{mask}}{\text{Threshold}_{critical}} \cdot (1 + \lambda \cdot \text{Eccentricity})\right)$$
   Penalizes both defect footprint and elongated crack morphology.
3. **Topological Depth Severity ($S_{depth}$)**:
   $$S_{depth} = \begin{cases}
   \min\left(1.0, \frac{|\Delta z|}{z_{max}}\right), & \text{if } \Delta z < -z_{nominal} \text{ (True structural cavity)} \\
   0.15 \cdot \min\left(1.0, \frac{|\Delta z|}{z_{max}}\right), & \text{if } |\Delta z| \le z_{nominal} \text{ (Superficial/Planar stain)}
   \end{cases}$$
4. **Operator Context Bias ($S_{op}$)**:
   $$S_{op} = \text{clamp}\left(\sum_j \omega_j \cdot F_j + \text{NLP\_Polarity}(Note), -0.3, +0.3\right)$$
   Where $F_j \in \{0, 1\}$ are operator checkbox flags (e.g., Tool Wear, Critical Seal).
5. **Composite Fused Severity**:
   $$S_{fused} = w_{vis} S_{vis} + w_{geo} S_{geo} + w_{depth} S_{depth} + w_{op} S_{op}$$
   With weights $w = [0.35, 0.20, 0.30, 0.15]$ constrained to $\sum w_i = 1.0$.

### 3.3 Uncertainty & Calibrated Risk Scoring (Stage 8)
1. **Aleatoric Uncertainty ($U_{aleatoric}$)**:
   $$U_{aleatoric} = \frac{-\sum_{k=1}^K p_k \ln(p_k)}{\ln(K)}$$
   Normalized Shannon entropy across the 6 defect classes ($U_{aleatoric} \in [0, 1]$).
2. **Epistemic Discrepancy ($U_{epistemic}$)**:
   $$U_{epistemic} = |S_{vis} - S_{depth}|$$
   Detects fundamental contradictions between visual prediction and physical depth relief.
3. **Calibrated Composite Risk Index ($R$)**:
   $$R = \text{clamp}\left(S_{fused} \cdot (1 + 0.2 \cdot U_{aleatoric}), 0.0, 1.0\right)$$

### 3.4 Decision Engine with Safety Hysteresis (Stage 9)
```
IF R >= 0.65 THEN:
    DISPOSITION = REJECT
    SEVERITY_LEVEL = CRITICAL
ELSE IF (R >= 0.25) OR (U_aleatoric > 0.45) OR (U_epistemic > 0.40) THEN:
    DISPOSITION = REVIEW
    SEVERITY_LEVEL = ELEVATED_UNCERTAINTY
ELSE:
    DISPOSITION = PASS
    SEVERITY_LEVEL = NOMINAL
END IF
```

---

## 4. API Specification & Data Contracts

### 4.1 `POST /api/v1/inspect`
Executes the full 10-stage inspection pipeline.

#### Request Payload:
```json
{
  "image_base64": "data:image/jpeg;base64,...",
  "component_type": "ANSI_B16_5_FLANGE",
  "serial_number": "FLG-2026-0881",
  "operator_input": {
    "operator_id": "OP-442",
    "notes": "Suspected tooling score near inner bore",
    "flags": {
      "tool_wear_alert": true,
      "critical_sealing_surface": true,
      "quenching_anomaly": false
    }
  },
  "config": {
    "enable_clahe": true,
    "clahe_clip_limit": 2.0,
    "threshold_pass": 0.25,
    "threshold_reject": 0.65,
    "uncertainty_tolerance": 0.40
  }
}
```

#### Response Payload:
```json
{
  "inspection_id": "INSP-6714-99201",
  "timestamp": "2026-09-09T19:54:38Z",
  "image_hash_sha256": "8f94d2e...b31",
  "summary": {
    "disposition": "REJECT",
    "risk_score": 0.884,
    "primary_defect": "Crack",
    "defect_confidence": 0.894,
    "uncertainty_score": 0.142
  },
  "pipeline_stages": {
    "stage_1_acquisition": {
      "dimensions": [1024, 1024, 3],
      "exposure_quality": "OPTIMAL"
    },
    "stage_2_preprocessing": {
      "clahe_applied": true,
      "denoise_method": "BILATERAL",
      "processed_image_base64": "data:image/png;base64,..."
    },
    "stage_3_zero_shot_clip": {
      "class_distribution": {
        "Crack": 0.894,
        "Dent": 0.052,
        "Scratch": 0.031,
        "Corrosion": 0.015,
        "Discoloration": 0.005,
        "Normal": 0.003
      },
      "heatmap_base64": "data:image/png;base64,...",
      "peak_coordinate": [355, 476],
      "bounding_box": [312, 405, 398, 547]
    },
    "stage_4_sam2_segmentation": {
      "mask_base64": "data:image/png;base64,...",
      "polygon_contour": [[312, 410], [330, 405], [395, 490], [380, 545], [320, 520]],
      "area_pixels": 4218,
      "area_mm2": 18.4,
      "iou_confidence": 0.942
    },
    "stage_5_depth_analysis": {
      "depth_map_base64": "data:image/png;base64,...",
      "relative_delta_z_mm": -1.82,
      "gradient_sharpness": 0.78,
      "topological_classification": "DEEP_CAVITY"
    },
    "stage_6_operator_context": {
      "sentiment_severity": 0.15,
      "notes_parsed": "tooling score near inner bore",
      "bias_applied": 0.12
    },
    "stage_7_evidence_fusion": {
      "weights": {"visual": 0.35, "geometric": 0.20, "depth": 0.30, "operator": 0.15},
      "components": {
        "visual_severity": 0.894,
        "geometric_severity": 0.844,
        "depth_severity": 0.910,
        "operator_severity": 0.700
      },
      "fused_score": 0.860
    },
    "stage_8_uncertainty": {
      "aleatoric_entropy": 0.142,
      "epistemic_discrepancy": 0.016,
      "calibrated_risk": 0.884
    },
    "stage_9_decision": {
      "disposition": "REJECT",
      "thresholds": {"pass": 0.25, "reject": 0.65},
      "safety_envelope_triggered": "HIGH_STRUCTURAL_RISK"
    },
    "stage_10_explainable_report": {
      "narrative": "Component rejected due to severe structural crack detected at coordinate (355, 476). CLIP visual identification indicated 89.4% confidence for fracture. SAM 2 delineated a 4,218 px irregular perimeter on the flange sealing face. Depth Anything V2 confirmed a -1.82 mm surface depression, ruling out superficial discoloration. Operator context corroborates mechanical fracture.",
      "audit_signature": "SIG-SHA256-88F9A1"
    }
  },
  "provenance": {
    "execution_mode": "MOCK_SIMULATION",
    "disclaimer": "SIMULATION ONLY: Generated by deterministic mock engine for testing.",
    "model_versions": {
      "clip": "MOCK-ViT-L/14",
      "sam2": "MOCK-SAM-2-Large",
      "depth": "MOCK-Depth-Anything-V2"
    }
  }
}
```

---

## 5. Realistic Mock AI Generation Engine

To ensure the prototype functions realistically out-of-the-box, the mock engine generates **mathematically accurate synthetic responses**:
1. **Specimen Catalog**:
   - Includes 6 pre-built high-resolution procedural industrial metal flange textures (Flange with hairline crack, radial scratch, deep dent, corrosion patch, heat discoloration, and pristine conforming flange).
2. **Procedural Heatmap Generation**:
   - Computes 2D bivariate Gaussian distributions centered at ground-truth defect coordinates with configurable covariance tensors $\Sigma = \begin{bmatrix} \sigma_x^2 & \sigma_{xy} \\ \sigma_{xy} & \sigma_y^2 \end{bmatrix}$ matching the defect aspect ratio.
   - Maps normalized intensity through the standard Jet / Turbo colormap.
3. **Procedural SAM 2 Polygon & Mask Generation**:
   - Generates jittered concave hulls bounding the defect region, producing realistic organic defect boundaries.
   - Rasterizes the hull into a binary mask PNG and vector coordinate list.
4. **Procedural Depth Topography Generation**:
   - Computes a toroidal/cylindrical base geometry representing the concentric flange face.
   - Injects a negative Gaussian pit for Cracks/Dents ($\Delta z < 0$) or flat zero-displacement ($\Delta z \approx 0$) for Discolorations.
   - Converts the depth matrix to Inferno pseudo-color.

---

## 6. Project Directory Structure

```
smartvision-oc/
├── docs/
│   ├── product.md            # Product Requirements & Scientific Specification
│   ├── ui.md                 # UI/UX Specification & Cockpit Design
│   └── engineering.md        # Technical Architecture & Engineering Spec
├── backend/
│   ├── main.py               # FastAPI entrypoint & router mounts
│   ├── config.py             # Environment config (AI_INFERENCE_BACKEND, etc.)
│   ├── schemas/              # Pydantic schemas (requests, responses, inspection)
│   ├── services/
│   │   ├── pipeline.py       # 10-Stage Pipeline Coordinator
│   │   ├── preprocessor.py   # CLAHE, Bilateral Denoising, Normalization
│   │   ├── fusion_engine.py  # Multi-Modal Evidence Fusion & Uncertainty
│   │   ├── decision_engine.py# Calibrated Risk Scoring & Hysteresis
│   │   ├── explainability.py # Natural language rationale generation
│   │   └── report_gen.py     # JSON & PDF audit report generation
│   ├── adapters/
│   │   ├── base.py           # Abstract Base Classes (BaseZeroShot, BaseSAM2, etc.)
│   │   ├── mock_adapter.py   # High-fidelity deterministic mock inference engine
│   │   ├── torch_clip.py     # PyTorch OpenCLIP implementation (optional plug-in)
│   │   ├── torch_sam2.py     # PyTorch SAM 2 implementation (optional plug-in)
│   │   └── torch_depth.py    # PyTorch Depth Anything V2 implementation (optional plug-in)
│   └── samples/              # Default synthetic flange specimens and ground truth metadata
├── frontend/
│   ├── index.html            # Vite entrypoint with high-contrast dark theme
│   ├── package.json          # React 18, Lucide Icons, Tailwind CSS, Canvas utilities
│   ├── tailwind.config.js    # Custom industrial dark palette (#0A0D14 base, etc.)
│   ├── src/
│   │   ├── App.jsx           # Main cockpit workstation layout
│   │   ├── components/
│   │   │   ├── Header.jsx           # Mode badge, flange selector, run action
│   │   │   ├── CanvasViewer.jsx     # Interactive multi-layer canvas with alpha blending
│   │   │   ├── StageStepper.jsx     # 10-stage visual progress tracker
│   │   │   ├── ControlPanel.jsx     # Image upload, CLAHE sliders, operator notes
│   │   │   ├── TelemetryPanel.jsx   # Class probabilities, depth profile, fusion radar
│   │   │   ├── DecisionCard.jsx     # PASS/REVIEW/REJECT status badge and risk meter
│   │   │   └── ReportModal.jsx      # ISO-style printable certificate drawer
│   │   ├── services/
│   │   │   ├── api.js               # Client API connector to backend
│   │   │   └── mockClient.js        # Standalone client-side mock fallback if offline
│   │   └── utils/
│   │       ├── canvasDrawing.js     # Vector polygon and heatmap blending logic
│   │       └── formatters.js        # Coordinate and metric formatters
└── tests/
    ├── test_fusion.py        # Unit tests for evidence fusion & risk calculation
    ├── test_decision.py      # Unit tests for hysteresis and threshold bounds
    └── test_api.py           # End-to-end API integration tests
```

---

## 7. Quality Assurance & Verification Plan

1. **Deterministic Test Suites**:
   - Verify that each defect specimen (Crack, Scratch, Dent, Corrosion, Discoloration, Normal) produces consistent, deterministic classifications and dispositions within the mock engine.
2. **Safety Hysteresis Verification**:
   - Ensure boundary values ($R = 0.25 \pm \epsilon$, $R = 0.65 \pm \epsilon$) cleanly resolve to the conservative status (`REVIEW` or `REJECT`) without fluttering.
3. **Pluggability Smoke Tests**:
   - Unit test the registry factory to confirm seamless dependency injection when toggling `AI_INFERENCE_BACKEND=MOCK` to `TORCH`.
4. **Frontend Canvas Performance**:
   - Benchmark 60 FPS canvas redraw rate during real-time alpha slider scrubbing and zoom/pan operations.
