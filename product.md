# Product Requirements & Scientific Specification
## SmartVisionOC: Explainable Multi-Modal AI Framework for Zero-Shot Industrial Quality Inspection

---

### Document Information
- **Project Name**: SmartVisionOC
- **Document Version**: 1.0.0
- **Document Type**: Product Requirements Document (PRD) & Scientific Specification
- **Target Domain**: Industrial Visual Quality Inspection (e.g., Precision Machined Metal Flanges, Castings, Aerospace/Automotive Components)
- **Author**: Lead AI Architect & Systems Engineering Team

---

## 1. Executive Summary & Value Proposition

### 1.1 The Industrial Challenge
Traditional automated optical inspection (AOI) and deep learning-based defect detection systems suffer from severe structural bottlenecks in real-world smart manufacturing:
1. **The Few-Shot / Cold-Start Problem**: Industrial defect rates are typically low (< 1-2%). Collecting and annotating thousands of supervised defect images for every new part variant, tooling revision, or casting batch takes weeks or months, causing costly production delays.
2. **Brittle Supervised Classifiers**: Conventional CNNs/YOLO models are closed-set detectors; they fail silently or misclassify rare, unmodeled, or novel defects (e.g., micro-voids, unexpected thermal discoloration, new chemical pitting).
3. **Black-Box Opacity**: High-stakes industries (automotive powertrain, aerospace turbines, nuclear pressure vessels) cannot deploy black-box models due to strict quality audit requirements (ISO 9001, AS9100, IATF 16949). Operators and quality engineers require explicit, justifiable causal explanations.
4. **Disconnection from Human Operators**: Shop-floor quality inspectors possess decades of contextual domain knowledge (e.g., "batch #44 had erratic furnace cooling", "tool wear observed at spindle 2"). Conventional AI systems cannot ingest this qualitative textual context.

### 1.2 The SmartVisionOC Solution
SmartVisionOC solves these challenges by pioneering a **Zero-Shot Multi-Modal Visual-Textual Inspection Framework** combining:
- **Zero-Shot Language-Supervised Recognition (CLIP)**: Open-vocabulary defect identification and coarse localization without prior task-specific defect training.
- **Promptable Foundation Segmentation (SAM 2)**: Precise pixel-level boundary delineation guided by visual semantic anchors.
- **Monocular 3D Spatial/Depth Topography (Depth Anything V2)**: Disambiguating 2D surface reflections/discolorations from true physical structural indentations/gouges.
- **Multi-Modal Evidence Fusion**: Rigorous mathematical fusion integrating visual confidence, geometric mask metrics, depth topography, and operator-entered contextual notes.
- **Transparent Decision Engine**: Calibrated risk scoring with configurable hysteresis ($PASS$, $REVIEW$, $REJECT$) and fully traceable natural language reasoning reports.

```
+----------------------------------------------------------------------------------------------------+
|                                    SMARTVISIONOC 10-STAGE PIPELINE                                  |
+----------------------------------------------------------------------------------------------------+
| [1. Image Acquisition]                                                                             |
|      High-resolution industrial capture of metal flange / machined part                            |
|             |                                                                                      |
| [2. Preprocessing & Enhancement]                                                                   |
|      CLAHE contrast enhancement, Bilateral filtering, illumination normalization                   |
|             |                                                                                      |
| [3. CLIP Zero-Shot Defect Identification & Coarse Localization]                                    |
|      Visual-language prompt alignment -> Defect class classification & spatial attention heatmap   |
|             |                                                                                      |
| [4. SAM 2 Precise Segmentation]                                                                    |
|      Coarse activation peak -> Point/box prompt -> Pixel-exact defect mask & area extraction       |
|             |                                                                                      |
| [5. Spatial & Depth Analysis (Depth Anything V2)]                                                  |
|      Monocular surface relative depth -> Profile depression (z-axis pit vs. 2D discoloration)       |
|             |                                                                                      |
| [6. Operator Input (Human-in-the-Loop)]                                                            |
|      Qualitative textual notes, shift context, specific tooling/lot suspicion                      |
|             |                                                                                      |
| [7. Multi-Modal Evidence Fusion]                                                                   |
|      Bayesian / Multi-factor combination of Visual, Geometric, Depth, and Operator evidence         |
|             |                                                                                      |
| [8. Uncertainty & Risk Scoring]                                                                    |
|      Epistemic & aleatoric uncertainty estimation -> Calibrated Risk Index R in [0, 1]              |
|             |                                                                                      |
| [9. Deterministic Decision Engine]                                                                 |
|      Tri-state classification with safety bounds: PASS | REVIEW | REJECT                           |
|             |                                                                                      |
| [10. Explainable Inspection Report]                                                                |
|      Audit-ready natural language rationale, multi-view overlays, downloadable inspection record   |
+----------------------------------------------------------------------------------------------------+
```

---

## 2. Personas & Stakeholders

| Persona | Role | Primary Goals | Key Pain Points Addressed |
| :--- | :--- | :--- | :--- |
| **Line Quality Inspector** | Shop-floor operator inspecting components at end-of-line | Rapid, reliable verification of parts; easily provide domain notes; view clear visual indicators. | No longer forced to make borderline calls alone; can input shift observations and see immediate fused risk score. |
| **Quality Assurance (QA) Engineer** | Establishes inspection criteria, tolerances, and calibration | Define pass/reject thresholds; audit model decisions; verify root causes of defects across batches. | Full explainability trace with metrics (area in $mm^2$, depth deviation, prompt similarity) eliminates black-box guesswork. |
| **Plant Operations Manager** | Responsible for throughput, yield, and scrap rates | Reduce false scrap rates; eliminate defective shipments to OEM clients; minimize setup time for new product lines. | Zero-shot capability means zero re-training downtime when introducing new flange geometries or alloy finishes. |
| **Regulatory & Compliance Auditor** | Audits manufacturing compliance (ISO 9001 / IATF 16949) | Verify that every automated rejection or pass decision is documented, reproducible, and verifiable. | Deterministic audit logs with image hash, timestamp, intermediate evidence, and operator sign-off. |

---

## 3. Detailed 10-Stage Pipeline Specification

### Stage 1: Image Acquisition
- **Input**: High-resolution RGB or monochromatic image of an industrial component (e.g., standard ANSI B16.5 forged steel weld-neck flange, CNC-milled face, or bearing race).
- **Supported Formats**: PNG, JPEG, TIFF, BMP, WebP.
- **Metadata Captured**: Image dimensions, capture timestamp, component serial number / Lot ID, optical exposure parameters.
- **Validation**: Dimension verification (minimum $512 \times 512$ px, recommended $1024 \times 1024$ px or higher), channel count check, blur/underexposure detection.

### Stage 2: Preprocessing and Enhancement
- **Purpose**: Normalize variable shop-floor lighting, mitigate specular metallic reflections, and accentuate high-frequency defect textures (hairline cracks, tool chattering, pitting).
- **Techniques**:
  1. **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: Amplifies subtle local surface contrast without over-amplifying noise in specular highlights. Tile grid: $8 \times 8$, Clip limit: $2.0$.
  2. **Edge-Preserving Denoising**: Bilateral filter ($\sigma_{color}=75$, $\sigma_{space}=75$) or median filter to smooth metal grain noise while preserving sharp crack boundaries.
  3. **Illumination Correction**: High-pass Gaussian filtering to neutralize uneven radial strobe lighting.
- **Output**: Enhanced inspection matrix $I_{enh} \in \mathbb{R}^{H \times W \times 3}$.

### Stage 3: CLIP Zero-Shot Defect Identification & Coarse Localization
- **Purpose**: Detect anomalous surface conditions without prior training by comparing image embeddings with prompt ensembles in a shared multi-modal embedding space.
- **Defect Taxonomy & Prompt Ensembles**:
  - `Crack`: *"a close-up industrial photo of a metal flange with a sharp fracture crack"*, *"hairline surface rupture in machined steel"*
  - `Scratch`: *"a close-up industrial photo of an abrasive linear scratch on metallic surface"*, *"machining tool score mark"*
  - `Dent`: *"a close-up industrial photo of an impact dent defect on a metal flange"*, *"concave mechanical indentation depression"*
  - `Corrosion`: *"a photo of rough oxidized surface corrosion and rust pitting on metal"*, *"chemical oxidation discoloration"*
  - `Discoloration`: *"a photo of uneven heat tint discoloration and thermal rainbow patina"*, *"surface oxidation tinting without depth deformation"*
  - `Normal`: *"a pristine flawless machined metal flange surface"*, *"conforming industrial metal component with smooth finish"*
- **Outputs**:
  - Normalized defect probability distribution $P_{CLIP} = [p_1, p_2, \dots, p_6]$ such that $\sum p_k = 1.0$.
  - Primary identified defect category $C^* = \arg\max_k (P_{CLIP, k})$.
  - Visual similarity heatmap $M_{sim} \in [0, 1]^{H \times W}$ derived via patch-token cross-attention or Grad-CAM over the visual transformer encoder.

### Stage 4: SAM 2 Precise Segmentation
- **Purpose**: Convert coarse visual attention peaks into crisp, sub-pixel accurate defect masks.
- **Mechanism**:
  - Extract the spatial peak $(x_{peak}, y_{peak}) = \arg\max_{(x,y)} M_{sim}(x,y)$ and bounding proposal box $[x_{min}, y_{min}, x_{max}, y_{max}]$ from the thresholded heatmap.
  - Dispatch prompt points and bounding boxes to the Segment Anything 2 (SAM 2) model.
  - Generate multi-scale candidate masks and select the highest IoU-confidence mask.
- **Outputs**:
  - Binary defect mask $M_{seg} \in \{0, 1\}^{H \times W}$.
  - Geometric metrics: Defect area in pixels, perimeter, bounding box coordinates, eccentricity, defect centroid $(x_c, y_c)$.

### Stage 5: Spatial and Depth Analysis (Depth Anything V2)
- **Purpose**: Disentangle purely visual 2D surface anomalies (such as benign oil marks, ink stamps, or mild cosmetic heat tint) from hazardous structural deformations (gouges, deep cracks, impacts).
- **Mechanism**:
  - Run monocular relative depth estimation across $I_{enh}$ yielding a continuous relative depth map $D(x, y) \in [0, 1]$.
  - Sample depth distribution over the defect mask $M_{seg}$ versus the surrounding local baseline flange surface $M_{base}$.
  - Calculate relative depth delta $\Delta z = \bar{D}_{defect} - \bar{D}_{base}$ and gradient sharpness $\nabla D$.
- **Outputs**:
  - Topographical depth map (rendered in pseudo-color colormap like Inferno or Viridis).
  - Depth classification: `Depression/Cavity` ($\Delta z < -0.15$), `Planar Surface Variation` ($|\Delta z| \le 0.15$), or `Protrusion/Burr` ($\Delta z > 0.15$).
  - Estimated depth severity index $S_{depth} \in [0, 1]$.

### Stage 6: Operator Input (Human-in-the-Loop)
- **Purpose**: Leverage human contextual intelligence to modulate the automated assessment.
- **Input Channels**:
  - **Structured Observations**: Checkboxes for known upstream issues (e.g., "Worn Tooling Warning", "Batch Quenching Irregularity", "Critical Flange Sealing Face").
  - **Unstructured Natural Language**: Free-form text input (e.g., *"Noticed recurring gouging near bolt hole 4 during turn 2"* or *"Cosmetic water stain from wash station, verify if depth is flat"*).
- **Processing**:
  - Text parsed via keyword/semantic intent matching for defect confirmation, severity modifiers, or false-positive mitigation cues.
- **Output**:
  - Operator confidence bias factor $O_{bias} \in [-0.3, +0.3]$ and operator severity weight $W_{op} \in [0, 1]$.

### Stage 7: Evidence Fusion
- **Purpose**: Mathematically synthesize heterogeneous visual, geometric, depth, and textual evidence streams into a unified evidence vector.
- **Fusion Formula**:
  The composite severity score $S_{fused}$ is computed via a weighted multi-criteria formulation:
  $$S_{fused} = w_{vis} \cdot P(C^*) + w_{geo} \cdot \Phi(Area, Ecc) + w_{depth} \cdot S_{depth} + w_{op} \cdot O_{bias}$$
  Where:
  - $w_{vis} = 0.35$ (CLIP visual confidence)
  - $w_{geo} = 0.20$ (Segmented defect size and geometric distortion)
  - $w_{depth} = 0.30$ (Physical 3D topological relief)
  - $w_{op} = 0.15$ (Human operator prior)
  - Subject to $\sum w_i = 1.0$.

### Stage 8: Uncertainty & Risk Scoring
- **Purpose**: Quantify confidence bounds to prevent overconfident automated rejections or missed hazardous failures.
- **Components**:
  - **Aleatoric Uncertainty** (Sensor/data ambiguity): Entropy of the CLIP prediction distribution $H(P) = -\sum p_k \log p_k$.
  - **Epistemic Uncertainty** (Model disagreement): Discrepancy between CLIP visual classification and Depth topology (e.g., CLIP detects "crack" but depth indicates completely flat planar surface).
- **Output**:
  - Calibrated Risk Score $R \in [0.00, 1.00]$.
  - Uncertainty metric $U \in [0.00, 1.00]$.

### Stage 9: Decision Engine
- **Purpose**: Render an automated, deterministic quality disposition with safety hysteresis.
- **Decision Logic**:
  $$\text{Decision} = \begin{cases}
  \text{PASS}, & \text{if } R < \theta_{pass} \text{ and } U < \theta_{uncert} \\
  \text{REJECT}, & \text{if } R \ge \theta_{reject} \\
  \text{REVIEW}, & \text{otherwise (moderate risk, high uncertainty, or boundary condition)}
  \end{cases}$$
- **Default Thresholds**:
  - $\theta_{pass} = 0.25$
  - $\theta_{reject} = 0.65$
  - $\theta_{uncert} = 0.40$

### Stage 10: Explainable Inspection Report
- **Purpose**: Provide a legally admissible, audit-compliant inspection breakdown for quality engineers and regulatory bodies.
- **Contents**:
  - Unique Inspection ID & Timestamp (ISO 8601).
  - Component details (Part Name, Serial Number, Image Hash SHA-256).
  - Executive Disposition Banner ($PASS$, $REVIEW$, $REJECT$) with Risk Index & Uncertainty.
  - Multi-Modal Visual Collage (Original, CLAHE Enhanced, CLIP Heatmap, SAM 2 Mask, Depth Topography).
  - Quantitative Evidence Table (Detected Class, Confidence %, Mask Area, Depth $\Delta z$, Operator Impact).
  - Natural Language Reasoning Paragraph (Auto-generated transparent narrative explaining why the decision was made).
  - Export capabilities (Downloadable JSON Data Package & Printable PDF/HTML Certificate).

---

## 4. Defect Taxonomy & Severity Definitions

| Defect Class | Physical Characteristics | Visual Indicators | Depth Profile Signature | Severity Weight ($\beta$) |
| :--- | :--- | :--- | :--- | :--- |
| **Crack** | Structural fracture, separation of material grain | Dark, jagged linear or branching lines | Sharp negative step ($\Delta z < -0.30$), steep $\nabla D$ | **Critical (1.00)** |
| **Dent** | Impact deformation, localized pit | Circular or elliptical shadow gradient | Smooth localized basin ($\Delta z < -0.20$) | **High (0.80)** |
| **Scratch** | Mechanical abrasive score or tool mark | Bright specular edges with dark centerline | Shallow linear trough ($-0.15 \le \Delta z < -0.05$) | **Moderate (0.50)** |
| **Corrosion** | Chemical oxidation, surface scaling, rust | Mottled, irregular chromatic texture (brown/orange) | Variable micro-roughness, shallow pitting | **High (0.75)** |
| **Discoloration** | Thermal heat tint, chemical stain, oil sheen | Chromatic shift (blue, gold, straw) with intact grain | Flat surface profile ($|\Delta z| < 0.05$) | **Low (0.25)** |
| **Normal** | Conforming machined finish, concentric tool marks | Uniform radial or cross-hatch reflection | Flat nominal baseline ($\Delta z \approx 0.00$) | **Zero (0.00)** |

---

## 5. Non-Functional & Operational Requirements

### 5.1 Pluggable AI Architecture Requirements
- **Simulated / Mock AI Inference**:
  - Must initially run on a deterministic synthetic simulation engine for all three foundation models (CLIP, SAM 2, Depth Anything V2).
  - The UI and API must explicitly declare `AI_MODE: "SIMULATED_MOCK"` without misleading the user that a GPU-backed deep neural network is executing.
  - The mock engine must produce deterministic, visually plausible results (synthetic heatmaps, clean polygon masks, and colorized depth gradients) matching the selected component and defect scenario.
- **Model Decoupling & Extensibility**:
  - The architecture must implement an Object-Oriented Adapter Pattern (`BaseZeroShotDetector`, `BaseSegmenter`, `BaseDepthAnalyzer`).
  - Swapping from Mock to real PyTorch/ONNX/HuggingFace inference must require zero frontend changes and only a configuration flag flip (`ENGINE_BACKEND=real`).

### 5.2 Performance & Responsiveness
- End-to-end pipeline execution time in Mock mode: **< 1.2 seconds**.
- UI rendering and canvas overlay toggle response time: **< 50 milliseconds**.
- Maximum file upload size: **25 MB**.

### 5.3 Reliability, Auditability & Safety
- **Reproducibility**: Identical inputs (image + operator text + configuration thresholds) must always produce identical risk scores and dispositions.
- **Audit Hash**: Every inspection run generates a cryptographic SHA-256 hash of the input image and parameters to guarantee anti-tamper compliance.
- **Fallback Behavior**: In the event of model ambiguity or processing timeouts, the system must fail-safe to `REVIEW` status, never silently defaulting to `PASS`.

---

## 6. Verification & Validation Metrics

1. **Classification Accuracy**: Top-1 zero-shot defect identification matching ground truth.
2. **Segmentation Quality**: Intersection-over-Union (IoU) between generated defect mask and annotated region $> 0.75$.
3. **Risk Calibration Error**: Brier score of calibrated risk index across validation splits $< 0.10$.
4. **False Rejection Rate (Type I error)**: $< 1.5\%$ on pristine conforming parts.
5. **False Acceptance Rate (Type II error)**: $< 0.01\%$ on critical structural defects (Cracks/Severe Dents).
