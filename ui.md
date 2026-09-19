# UI/UX Design Specification & Visual Architecture
## SmartVisionOC: Industrial AI Quality Inspection Dashboard

---

### Document Information
- **Document Version**: 1.0.0
- **Document Type**: UI/UX Specification & Design System
- **Target Interface**: Desktop / Industrial Tablet Cockpit (Minimum $1440 \times 900$, Optimized for $1920 \times 1080$ Full HD & $4K$ Shop-floor Monitors)
- **Aesthetic Benchmark**: Aerospace & Semiconductor Metrology Systems (KLA, Cognex, Keyence, W&B, Roboflow Industrial)

---

## 1. Design Philosophy & Aesthetic Guidelines

### 1.1 Core Principles
1. **High Information Density with Zero Clutter**: Shop-floor quality engineers require comprehensive telemetry (defect coordinates, confidence curves, depth delta, risk indices) at a glance without pagination or excessive modal switching.
2. **Industrial Dark Mode ("Cockpit" Theme)**: Eliminates eye strain under harsh factory floor fluorescent lighting; provides high contrast for fluorescent defect heatmaps, segmentation outlines, and pseudo-color depth maps.
3. **Fail-Safe Visual Signatures**: Inspection outcomes ($PASS$, $REVIEW$, $REJECT$) use distinct, unmissable color coding, redundant icon symbology, and high-visibility state badges for colorblind accessibility.
4. **Transparent AI Reality Principle**: The system explicitly and conspicuously displays its operational mode (`[MOCK / SIMULATED INFERENCE]` vs. `[LIVE GPU INFERENCE]`), ensuring full compliance with industrial validation standards.

### 1.2 Color Palette

```
+----------------------------------------------------------------------------------------------------+
| COLOR ROLE                | HEX CODE  | RGB              | USAGE                                   |
+---------------------------+-----------+------------------+-----------------------------------------+
| Background Base           | #0A0D14   | rgb(10, 13, 20)  | App root, canvas surround               |
| Panel Surface             | #111726   | rgb(17, 23, 38)  | Cards, sidebars, inspection modules     |
| Elevated Surface / Hover  | #1B2438   | rgb(27, 36, 56)  | Card headers, hover states, active tabs |
| Border Subdued            | #26334D   | rgb(38, 51, 77)  | Panel dividers, grid lines              |
| Border Highlight          | #3B82F6   | rgb(59, 130, 246)| Focused inputs, active overlay layers   |
| Primary Text              | #F8FAFC   | rgb(248, 250, 252| High-contrast titles, metrics, values   |
| Secondary Text            | #94A3B8   | rgb(148, 163, 184| Labels, units, metadata timestamps      |
| Muted Text                | #475569   | rgb(71, 85, 105) | Inactive toggles, subtle footnotes      |
| Pass / Nominal            | #10B981   | rgb(16, 185, 129)| PASS disposition, conforming surface    |
| Review / Caution          | #F59E0B   | rgb(245, 158, 11)| REVIEW disposition, high uncertainty    |
| Reject / Critical         | #EF4444   | rgb(239, 68, 68) | REJECT disposition, severe fracture/dent|
| Cyber Cyan (SAM 2 Mask)   | #06B6D4   | rgb(6, 182, 212) | Segmentation polygons, contour bounds   |
| Electric Amber (Heatmap)  | #F97316   | rgb(249, 115, 22)| CLIP Grad-CAM activation peak           |
| Neon Purple (Depth Map)   | #8B5CF6   | rgb(139, 92, 246)| Depth topographical contours & 3D relief|
+----------------------------------------------------------------------------------------------------+
```

### 1.3 Typography
- **Primary Interface Font**: `Inter`, `-apple-system`, `sans-serif` (Optimal legibility for UI controls, buttons, tooltips).
- **Technical & Telemetry Monospace Font**: `JetBrains Mono`, `Fira Code`, `Consolas` (Used for coordinates, bounding boxes, hex hashes, risk percentages, and confidence metrics).
- **Type Scale**:
  - `Display / Status Banner`: $28\text{px} - 32\text{px}$ (Heavy Bold, uppercase tracking +1.5px)
  - `Header Section`: $16\text{px} - 18\text{px}$ (Semibold)
  - `Body / Descriptions`: $13\text{px} - 14\text{px}$ (Regular)
  - `Telemetry / Data Metrics`: $12\text{px} - 13\text{px}$ (Monospace Medium)
  - `Micro Badges & Legends`: $10\text{px} - 11\text{px}$ (Monospace Bold, uppercase)

---

## 2. Comprehensive Layout Architecture

The user interface follows a **Tri-Column Cockpit Grid** anchored by a persistent Global Header and a collapsible bottom-docked Inspection Audit Drawer.

```
+------------------------------------------------------------------------------------------------------------------------+
| GLOBAL SYSTEM HEADER                                                                                                   |
| [LOGO] SmartVisionOC v0.1.0-proto | [STATUS: SIMULATED AI ENGINE] | Flange Sample Selector | Batch #Lot-9021 | [RUN] [REPORT]  |
+------------------------------+---------------------------------------------------------+-------------------------------+
| COLUMN 1: CONTROLS & INPUTS  | COLUMN 2: MULTI-MODAL VISION WORKSTATION               | COLUMN 3: TELEMETRY & DECISION|
| (Width: 320px - 360px)       | (Flexible Center: ~800px - 1100px)                      | (Width: 380px - 420px)        |
|                              |                                                         |                               |
| 1. Sample Selector / Upload  | 1. Stage Tabs:                                          | 1. Primary Disposition Card:  |
|    - Drag & Drop Dropzone    |    [All (2x2)] [Raw] [CLAHE] [CLIP] [SAM 2] [Depth]     |    +------------------------+ |
|    - Preset Flange Scenarios |                                                         |    |      REJECT (94.2%)    | |
|      * Hairline Crack        | 2. Interactive High-Res Canvas:                         |    |   Risk Score: 0.88/1.0 | |
|      * Tooling Scratch       |    +--------------------------------------------------+ |    +------------------------+ |
|      * Severe Impact Dent    |    |                                                  | |                               |
|      * Chemical Corrosion    |    |             [Visual Inspection Canvas]           | | 2. Zero-Shot Class Spectrum: |
|      * Heat Discoloration    |    |       - Bounding Box overlay                     | |    Crack        [=======] 89%|
|      * Pristine / Normal     |    |       - High-precision SAM 2 polygon             | |    Dent         [==     ] 18%|
|                              |    |       - Dynamic alpha blend slider               | |    Scratch      [=      ] 06%|
| 2. Preprocessing Config:     |    |                                                  | |    Corrosion    [       ] 02%|
|    [v] CLAHE Normalization   |    +--------------------------------------------------+ |    Normal       [       ] 01%|
|    [v] Bilateral Denoising   |                                                         |                               |
|                              | 3. Canvas Quick Controls:                               | 3. Depth & Topography Card:   |
| 3. Operator Context Input:   |    [Reset Zoom] [Fit] [100%] [Toggle Crosshair]          |    - Defect Depth: -1.82 mm   |
|    - Upstream Checkboxes     |    Layer Alpha Slider: [======O=======] 65%             |    - Topology: Deep Cavity    |
|    - Operator Notes Textbox  |                                                         |                               |
|    - Shift / Inspector ID    | 4. 10-Stage Pipeline Status Stepper:                    | 4. Evidence Fusion Breakdown: |
|                              |    (1)->(2)->(3)->(4)->(5)->(6)->(7)->(8)->(9)->(10)   |    Visual: 35% | Depth: 30%   |
| [Execute Inspection Button]  |                                                         |    Geo: 20%    | Oper: 15%    |
|                              |                                                         |                               |
|                              |                                                         | 5. Natural Language Rationale |
|                              |                                                         |    "Critical structural crack |
|                              |                                                         |     detected near bolt 3..."  |
+------------------------------+---------------------------------------------------------+-------------------------------+
| BOTTOM DOCK: EXPANDABLE AUDIT LOG & CERTIFICATE PREVIEW (COLLAPSIBLE DRAWER)                                           |
| [^] Open Full ISO-Compliant Inspection Certificate | SHA-256: 8f94d...3b | Download JSON | Export Printable PDF         |
+------------------------------------------------------------------------------------------------------------------------+
```

---

## 3. Component Deep Dive

### 3.1 Global System Header
- **Branding Area**: `SmartVisionOC` in bold sans-serif with a glowing cyan optical reticle icon. Subtitle: `Zero-Shot Industrial Inspection`.
- **Inference Mode Indicator**:
  - `[SIMULATED AI INFERENCE]`: Amber pill badge with pulsing dot and tooltip: *"Running on deterministic architectural mock engine. Ready for PyTorch/ONNX backend."*
  - Allows one-click switching to `[LIVE PYTORCH (CUDA)]` if a real backend is detected.
- **Quick Preset Selector**: Dropdown to instantly load pre-configured industrial metal flange test specimens with known ground truth defect types:
  1. `Specimen #FLG-CRK-01`: Machined Flange with Hairline Fracture Crack
  2. `Specimen #FLG-SCR-04`: Gasket Sealing Face with Concentric Scratch
  3. `Specimen #FLG-DNT-09`: Bolt Lug with Heavy Impact Dent
  4. `Specimen #FLG-COR-12`: Flange Web with Chemical Oxidation/Pitting
  5. `Specimen #FLG-DSC-15`: Thermal Heat Tint / Temper Coloration
  6. `Specimen #FLG-NRM-00`: Conforming Pristine ANSI Class 300 Flange
- **Header Actions**:
  - `Run Full Pipeline (Space)` button with glowing border.
  - `Reset All` button.
  - `Export Report (Ctrl+E)` button.

### 3.2 Column 1: Inputs & Operator Control Panel
- **Module 1: Industrial Image Dropzone**:
  - Drag-and-drop zone supporting drag over with visual glowing border.
  - Displays preview thumbnail, file size, dimensions ($W \times H$), and calculated SHA-256 hash.
- **Module 2: Preprocessing Parameters (Stage 2)**:
  - Toggle: `CLAHE Contrast Enhancement` (Enabled by default).
  - Slider: `Clip Limit` ($1.0$ to $4.0$, default $2.0$).
  - Toggle: `Bilateral Denoising` (Preserves step edges while attenuating alloy grain).
  - Toggle: `Illumination Flattening` (Normalizes radial vignette shadows).
- **Module 3: Zero-Shot Prompt Engineering (Stage 3)**:
  - Expandable accordion allowing the quality engineer to inspect or edit the natural language prompts fed into CLIP.
  - Class threshold sensitivity slider ($\theta_{defect} \in [0.10, 0.90]$).
- **Module 4: Operator Context & Human-in-the-Loop (Stage 6)**:
  - **Operator Notes**: Textarea with placeholder: *"Enter shop-floor observations, e.g. 'Abnormal vibration heard during finish milling, inspect outer rim near hole 2'..."*
  - **Contextual Flags**:
    - `[ ] Tool Wear Alert (End of Tool Life)`
    - `[ ] Critical High-Pressure Sealing Surface`
    - `[ ] Known Upstream Heat-Treatment Anomaly`
  - Dynamic indicator showing the real-time weight modification that operator input contributes to the Evidence Fusion engine.

### 3.3 Column 2: Multi-Modal Vision Workstation
- **View Mode Switcher Bar**:
  - `Quad View (2x2)`: Displays Raw Image, CLAHE Enhanced, CLIP Heatmap, and SAM 2 Mask simultaneously for instant comparative inspection.
  - `Original Image`: Raw pristine sensor feed with defect bounding box.
  - `CLAHE Enhanced`: Normalized surface textures.
  - `CLIP Heatmap`: Continuous temperature gradient colormap (Jet / Turbo) highlighting zero-shot coarse localization.
  - `SAM 2 Mask`: Crisp cyan polygon overlay showing pixel-level segmented boundary.
  - `Depth Anything V2 Topography`: Pseudo-color elevation map (Inferno/Viridis) showing topological depression vs. elevation.
  - `Composite Fusion`: Multi-layer composite with original image background, translucent heatmap, vector boundary lines, and depth contour rings.
- **Overlay Control Bar**:
  - Layer Alpha Slider: Real-time slider ($0\%$ to $100\%$) blending the selected overlay onto the original metal surface.
  - Coordinate Readout: Hovering over the canvas displays real-time pixel coordinates `X: 428, Y: 612`, localized pixel intensity, and depth $Z$-value.
  - Interactive Point Prompting: Ability to click anywhere on the canvas to manually inject a positive/negative prompt point for SAM 2 segmentation refinement.
- **10-Stage Pipeline Stepper**:
  - Horizontal progress bar at the bottom of the canvas showing the 10 sequential pipeline stages with micro-status indicators (Completed, In-Progress, Pending, or Skipped).
  - Clicking any stage jumps the telemetry display to that stage's specific diagnostic outputs.

### 3.4 Column 3: Telemetry, Fusion & Decision Engine
- **Module 1: Quality Disposition Card (Stage 9)**:
  - Prominent status box:
    - If `PASS`: Emerald green background with checkmark icon and text: `PASS - COMPONENT CONFORMING`.
    - If `REVIEW`: Amber background with alert triangle and text: `REVIEW - OPERATOR VERIFICATION REQUIRED`.
    - If `REJECT`: Crimson red background with octagonal stop icon and text: `REJECT - NON-CONFORMING DEFECT DETECTED`.
  - Calibrated Risk Gauge ($R \in [0.00, 1.00]$) with dynamic threshold markers ($\theta_{pass}=0.25$, $\theta_{reject}=0.65$).
  - Uncertainty Bar: Aleatoric vs. Epistemic uncertainty percentage.
- **Module 2: Zero-Shot Defect Class Distribution (Stage 3)**:
  - Real-time horizontal bar graph displaying normalized softmax confidence for all six defect categories (`Crack`, `Dent`, `Scratch`, `Corrosion`, `Discoloration`, `Normal`).
  - Top identified class highlighted with badge.
- **Module 3: Spatial & Depth Topography Metrics (Stage 4 & 5)**:
  - Geometric Profile:
    - Bounding Box: `[X: 312, Y: 405, W: 86, H: 142]` px
    - Segmented Mask Area: `4,218 px` ($\approx 18.4 \text{ mm}^2$)
    - Aspect Ratio & Perimeter: `1.65` / `312 px`
  - Depth Profile:
    - Relative Surface Displacement $\Delta z$: `-1.82 mm` (Cavity/Pit)
    - Surface Gradient $\nabla D$: High steepness ($0.78$)
    - 3D Profile Classification: `True Structural Cavity` (Disproves surface-only cosmetic stain)
- **Module 4: Evidence Fusion Balance (Stage 7 & 8)**:
  - Radar or stacked contribution bar displaying:
    - Visual Severity ($35\%$ weight)
    - Geometric Severity ($20\%$ weight)
    - Depth Topography Severity ($30\%$ weight)
    - Operator Context Modifier ($15\%$ weight)
- **Module 5: Natural Language Explainability Narrative (Stage 10)**:
  - Auto-generated transparent rationale synthesized by the explainability engine:
    > *"Component rejected due to severe localized material fracture detected at coordinate (355, 476). CLIP visual identification indicated 89.4% confidence for structural crack. SAM 2 segmentation delineated a 4,218 px irregular perimeter on the flange sealing face. Depth Anything V2 confirmed a -1.82 mm surface depression, ruling out superficial discoloration. Operator observation regarding abnormal milling vibration corroborates mechanical fracture."*

---

## 4. Inspection Report & Export View (Stage 10 Modal / Drawer)

### 4.1 Printable Certificate Layout
When the user clicks `Export Report` or opens the full report drawer:
1. **Header**: ISO 9001 / IATF 16949 Inspection Certificate styling.
2. **Metadata Grid**: Serial number, Part Description, Inspection Timestamp, Inspector Name/ID, Workstation ID, SHA-256 Image Hash.
3. **Executive Summary**: Big color-coded disposition stamp ($PASS$ / $REVIEW$ / $REJECT$), Final Risk Score, Risk Thresholds.
4. **Visual Evidence Gallery**: Side-by-side high-resolution renders of:
   - Raw Captured Image
   - Preprocessed Contrast View
   - CLIP Localization Heatmap
   - SAM 2 Precise Segmentation Mask
   - Depth Anything V2 Elevation Topography
5. **Detailed Metric Tables**: Tabulated numerical measurements (Mask Area in $mm^2$, Depth depression in $mm$, Class Confidences).
6. **Operator Context & Causal Analysis**: Verbatim operator notes, upstream flags, and the automated natural language synthesis.
7. **Sign-off Block**: Inspector signature placeholder, supervisor sign-off, date, and audit trail record.

### 4.2 Export Formats
- **Interactive Web Preview**: Clean drawer sliding up from the bottom of the screen.
- **Export to JSON**: Complete structured data dump including all coordinates, vectors, bounding boxes, weights, and timestamps.
- **Print / PDF Generator**: Styled `@media print` CSS layout that produces a clean, single-page or two-page executive PDF certificate.

---

## 5. Interaction Patterns & State Transitions

### 5.1 Pipeline Execution Lifecycle
1. **IDLE**: Specimen loaded, parameters configured. Canvas displays pristine raw image. "Run Inspection" button pulses.
2. **ACQUIRING & PREPROCESSING (0-200ms)**: Loading skeleton shimmer across canvas. CLAHE filter executes.
3. **ZERO-SHOT CLIP INFERENCE (200-500ms)**: Heatmap layer renders with subtle fade-in. Confidence bars animate from left to right.
4. **SAM 2 SEGMENTATION (500-800ms)**: Bounding box snaps into position; cyan polygon mask boundary animates with glowing stroke.
5. **DEPTH TOPOGRAPHY ESTIMATION (800-1000ms)**: Pseudo-color depth relief renders; $\Delta z$ metric calculates.
6. **EVIDENCE FUSION & DECISION (1000-1200ms)**: Risk meter sweeps to calculated value. Primary disposition card illuminates in green, amber, or red with sound/haptic visual cues.
7. **EXPLAINABILITY REPORT READY**: Report button activates; natural language paragraph renders with typing effect or smooth fade.

### 5.2 Responsive & Touch-Friendly Adaptations
- Layout gracefully wraps into a 2-column or stacked 1-column view for shop-floor industrial tablets (e.g., 10-inch Panasonic Toughbook or iPad Pro).
- Interactive touch targets: minimum $48 \times 48$ px for buttons, sliders, and layer toggles to accommodate gloved operation on the factory floor.
- Keyboard shortcuts:
  - `Space`: Execute full inspection pipeline.
  - `1` through `6`: Switch canvas view modes (Raw, CLAHE, CLIP, SAM 2, Depth, Composite).
  - `R`: Toggle Inspection Report modal.
  - `O`: Toggle overlay visibility on/off.
