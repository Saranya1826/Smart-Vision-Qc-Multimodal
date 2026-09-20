# SmartVisionOC: Explainable Multi-Modal AI Framework for Zero-Shot Industrial Quality Inspection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**SmartVisionOC** is an industrial-grade visual quality inspection framework engineered for zero-shot defect identification, precise segmentation metrology, spatial depth analysis, and explainable decision auditing in precision manufactured metal components (such as ANSI/ASME B16.5 pipe flanges, turned shafts, bearings, and machined castings).

---

## 🚀 Key Features

* **10-Stage Optical Metrology Pipeline**: An end-to-end sequential inspection workflow spanning acquisition, enhancement, identification, segmentation, 3D topography, evidence fusion, and automated reporting.
* **Zero-Shot Defect Identification**: Standardized 11-class industrial defect taxonomy (`Normal`, `Crack`, `Scratch`, `Dent`, `Corrosion`, `Hole`, `Surface Damage`, `Missing Part`, `Deformation`, `Contamination`, `Other Anomaly`).
* **Multi-Modal Evidence Fusion**: Mathematically synthesizes visual embeddings ($35\%$), segmentation metrology ($30\%$), 3D depth relief ($20\%$), and shop-floor operator context ($15\%$) with contradiction damping.
* **Calibrated Risk Engine**: Continuous $[0.00, 1.00]$ risk scoring with configurable industrial hysteresis thresholds (`PASS` $< 0.25$, `REVIEW` $0.25 - 0.65$, `REJECT` $\ge 0.65$).
* **Automated Macro-Geometry Filtering**: Automatically isolates background silhouettes and intentional machined through-holes (bolt holes, bores, chamfers) to prevent false-positive defect attribution on conforming parts.
* **Pluggable Architecture**: Seamlessly transitions between production GPU neural weights (PyTorch CLIP ViT-B/32, SAM 2, Depth Anything V2) and an authentic Computer Vision Metrology Fallback Engine.
* **ISO 9001 / IATF 16949 Audit Trail**: Synthesizes natural-language causal reasoning, actionable shop-floor directives, printable inspection certificates, and downloadable JSON audit packages.

---

## 🏗️ 10-Stage Pipeline Architecture

```
1. Image Acquisition ──► 2. Preprocessing & QA ──► 3. Defect Identification (CLIP)
                                                           │
6. Operator HITL ───────► 7. Evidence Fusion ◄── 4. Precise Segmentation (SAM 2)
                                 │                         │
                                 ▼                5. Spatial Depth Topography
                         8. Risk Scoring
                                 │
                                 ▼
                         9. Decision Engine ──► 10. Explainable Report
```

---

## 🛠️ Quick Start

### 1. Prerequisites
- Python 3.10 or higher
- Modern web browser (Chrome, Edge, Firefox)

### 2. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/Saranya1826/SmartVisionOC.git
cd SmartVisionOC
pip install -r requirements.txt
```

*(Or install core dependencies directly)*:
```bash
pip install fastapi uvicorn pydantic opencv-python pillow numpy scipy
```

### 3. Launching SmartVisionOC

You can run SmartVisionOC in two modes:

#### Option A: Native Windows Desktop App (Recommended)
Launch the application as a standalone desktop app (without browser chrome or address bars) using Microsoft Edge Chromium WebView2:
- **One-Click**: Double-click [`SmartVisionOC.bat`](file:///C:/Users/Hxtreme/Documents/paper_publication/Smart%20Vision%20Qc-Multimodal/SmartVisionOC.bat)
- **Command Line**:
  ```bash
  python desktop.py
  ```

#### Option B: Web Browser Dashboard
Start the local FastAPI inspection server and open in your favorite web browser:
```bash
python run.py
```
Open your browser and navigate to:
```
http://127.0.0.1:8000
```

---

## 🧪 Verification & Automated Tests
Execute the verification test suite covering preset specimens, user uploads, state isolation, and macro-geometry validation:
```bash
python tests/test_pipeline.py
```

---

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.
