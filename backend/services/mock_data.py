"""
Synthetic Industrial Flange Generator & Mock Specimen Database
Provides realistic, deterministic industrial metal flange images and ground-truth defect parameters.
"""
import io
import base64
import math
import numpy as np
import cv2
from PIL import Image
from typing import Dict, Any, List, Tuple

SPECIMEN_CATALOG = {
    "FLG-CRK-01": {
        "id": "FLG-CRK-01",
        "name": "Specimen #FLG-CRK-01 (Hairline Structural Crack)",
        "defect_type": "Crack",
        "ground_truth_severity": "Critical",
        "description": "ANSI B16.5 Class 300 Flange with severe 48mm hairline fracture propagating from outer bolt hole 3 towards inner raised sealing face.",
        "defect_params": {
            "center": (360, 480),
            "size": (45, 130),
            "angle": 35,
            "type": "crack",
            "delta_z_mm": -1.85,
            "profile": "Structural Cavity",
            "base_conf": 0.894,
            "location_name": "Raised Sealing Face adjacent to Bolt Bore #3"
        }
    },
    "FLG-SCR-04": {
        "id": "FLG-SCR-04",
        "name": "Specimen #FLG-SCR-04 (Concentric Tooling Scratch)",
        "defect_type": "Scratch",
        "ground_truth_severity": "Moderate",
        "description": "Radial tooling score along gasket seating ring caused by chipped ceramic turning insert.",
        "defect_params": {
            "center": (512, 330),
            "size": (180, 20),
            "angle": -15,
            "type": "scratch",
            "delta_z_mm": -0.35,
            "profile": "Shallow Surface Trough",
            "base_conf": 0.842,
            "location_name": "Concentric Gasket Retaining Ring"
        }
    },
    "FLG-DNT-09": {
        "id": "FLG-DNT-09",
        "name": "Specimen #FLG-DNT-09 (Mechanical Impact Dent)",
        "defect_type": "Dent",
        "ground_truth_severity": "High",
        "description": "Plastic deformation depression on bevel edge caused by material handling drop impact.",
        "defect_params": {
            "center": (620, 580),
            "size": (75, 65),
            "angle": 10,
            "type": "dent",
            "delta_z_mm": -1.42,
            "profile": "Impact Basin Cavity",
            "base_conf": 0.881,
            "location_name": "Outer Chamfer Rim at 5 O'clock"
        }
    },
    "FLG-COR-12": {
        "id": "FLG-COR-12",
        "name": "Specimen #FLG-COR-12 (Chemical Oxidation / Pitting)",
        "defect_type": "Corrosion",
        "ground_truth_severity": "High",
        "description": "Mottled ferric oxidation cluster with intergranular pitting along flange neck weld preparation.",
        "defect_params": {
            "center": (420, 390),
            "size": (90, 85),
            "angle": 0,
            "type": "corrosion",
            "delta_z_mm": -0.75,
            "profile": "Eroded Pitted Micro-Roughness",
            "base_conf": 0.865,
            "location_name": "Weld Neck Transition Zone"
        }
    },
    "FLG-DSC-15": {
        "id": "FLG-DSC-15",
        "name": "Specimen #FLG-DSC-15 (Thermal Heat Discoloration)",
        "defect_type": "Discoloration",
        "ground_truth_severity": "Low",
        "description": "Straw-to-blue thermal oxidation patina from torch pre-heating. Nominal thickness intact, no depth deformation.",
        "defect_params": {
            "center": (530, 490),
            "size": (110, 100),
            "angle": 25,
            "type": "discoloration",
            "delta_z_mm": -0.02,
            "profile": "Planar Surface Stain (Flat)",
            "base_conf": 0.812,
            "location_name": "Inner Machined Bore Landing"
        }
    },
    "FLG-NRM-00": {
        "id": "FLG-NRM-00",
        "name": "Specimen #FLG-NRM-00 (Pristine Conforming Flange)",
        "defect_type": "Normal",
        "ground_truth_severity": "Nominal",
        "description": "Conforming ASTM A105 forged carbon steel flange. Uniform 63 micro-inch Ra phonographic gramophone finish. Zero anomalies.",
        "defect_params": {
            "center": (512, 512),
            "size": (0, 0),
            "angle": 0,
            "type": "normal",
            "delta_z_mm": 0.00,
            "profile": "Conforming Baseline Face",
            "base_conf": 0.965,
            "location_name": "Entire Surface Nominal"
        }
    }
}

def generate_procedural_flange(specimen_id: str, width: int = 1024, height: int = 1024) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Generates a photorealistic synthetic industrial metal flange image
    with precision CNC lathe marks, bolt holes, metallic grain, and deterministic defect injection.
    """
    specimen = SPECIMEN_CATALOG.get(specimen_id, SPECIMEN_CATALOG["FLG-CRK-01"])
    params = specimen["defect_params"]
    
    # Base metallic backdrop (dark industrial inspection booth surface)
    img = np.full((height, width, 3), 32, dtype=np.uint8)
    
    center_x, center_y = width // 2, height // 2
    r_outer = int(width * 0.44)
    r_bolt_circle = int(width * 0.35)
    r_raised_face = int(width * 0.28)
    r_bore = int(width * 0.15)
    
    # 1. Draw outer flange body (machined steel)
    cv2.circle(img, (center_x, center_y), r_outer, (150, 155, 160), -1, lineType=cv2.LINE_AA)
    
    # Concentric lathe toolpaths / gramophone spiral phonographic finish
    for r in range(r_bore, r_outer, 2):
        shade = int(140 + 25 * math.sin(r * 0.4) + (r % 7) * 2)
        cv2.circle(img, (center_x, center_y), r, (shade - 8, shade, shade + 5), 1, lineType=cv2.LINE_AA)
        
    # 2. Draw Raised Face ring
    cv2.circle(img, (center_x, center_y), r_raised_face, (168, 172, 176), -1, lineType=cv2.LINE_AA)
    for r in range(r_bore, r_raised_face, 2):
        shade = int(155 + 20 * math.cos(r * 0.3) + (r % 5) * 3)
        cv2.circle(img, (center_x, center_y), r, (shade - 5, shade, shade + 8), 1, lineType=cv2.LINE_AA)
        
    # 3. Draw Bolt Holes (8 standard bolt pattern)
    num_bolts = 8
    bolt_radius = int(width * 0.045)
    for i in range(num_bolts):
        theta = (2 * math.pi / num_bolts) * i
        bx = int(center_x + r_bolt_circle * math.cos(theta))
        by = int(center_y + r_bolt_circle * math.sin(theta))
        # Inner dark hole
        cv2.circle(img, (bx, by), bolt_radius, (20, 22, 25), -1, lineType=cv2.LINE_AA)
        # Chamfer highlight
        cv2.circle(img, (bx, by), bolt_radius + 2, (200, 205, 210), 1, lineType=cv2.LINE_AA)
        cv2.circle(img, (bx, by), bolt_radius + 4, (90, 95, 100), 1, lineType=cv2.LINE_AA)
        
    # 4. Draw Center Bore
    cv2.circle(img, (center_x, center_y), r_bore, (15, 16, 18), -1, lineType=cv2.LINE_AA)
    cv2.circle(img, (center_x, center_y), r_bore + 3, (215, 220, 225), 1, lineType=cv2.LINE_AA)
    
    # 5. Add subtle metallic speckle noise & anisotropic brushed sheen
    noise = np.random.normal(0, 7, (height, width, 3)).astype(np.int16)
    img_with_noise = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # 6. Inject Defect
    dtype = params["type"]
    cx, cy = params["center"]
    w, h = params["size"]
    
    if dtype == "crack":
        # Jagged branching fracture
        pts = []
        curr_x, curr_y = cx - w // 2, cy - h // 2
        pts.append([curr_x, curr_y])
        steps = 14
        for s in range(steps):
            curr_x += int((w / steps) + np.random.randint(-4, 6))
            curr_y += int((h / steps) + np.random.randint(-8, 9))
            pts.append([curr_x, curr_y])
        pts = np.array(pts, np.int32)
        # Deep fracture groove
        cv2.polylines(img_with_noise, [pts], False, (18, 20, 22), 5, lineType=cv2.LINE_AA)
        # Sharp specular stress edge
        cv2.polylines(img_with_noise, [pts + 2], False, (220, 230, 240), 1, lineType=cv2.LINE_AA)
        
    elif dtype == "scratch":
        # Concentric or linear abrasive score
        pts = []
        for a in range(-w // 2, w // 2, 8):
            py = int(cy + math.sin(a * 0.05) * 8 + np.random.randint(-2, 3))
            px = cx + a
            pts.append([px, py])
        pts = np.array(pts, np.int32)
        cv2.polylines(img_with_noise, [pts], False, (40, 45, 50), 3, lineType=cv2.LINE_AA)
        cv2.polylines(img_with_noise, [pts - 1], False, (230, 240, 250), 1, lineType=cv2.LINE_AA)
        
    elif dtype == "dent":
        # Mechanical impact pit with shadow and specular lip
        cv2.ellipse(img_with_noise, (cx, cy), (w // 2, h // 2), params["angle"], 0, 360, (50, 52, 55), -1, lineType=cv2.LINE_AA)
        cv2.ellipse(img_with_noise, (cx - 4, cy - 4), (w // 2 - 8, h // 2 - 8), params["angle"], 0, 360, (25, 27, 30), -1, lineType=cv2.LINE_AA)
        cv2.ellipse(img_with_noise, (cx + 8, cy + 8), (w // 2 - 4, h // 2 - 4), params["angle"], 0, 180, (220, 225, 230), 2, lineType=cv2.LINE_AA)
        
    elif dtype == "corrosion":
        # Mottled ferric oxidation cluster
        for _ in range(40):
            rx = cx + np.random.randint(-w // 2, w // 2)
            ry = cy + np.random.randint(-h // 2, h // 2)
            rad = np.random.randint(4, 18)
            col = (np.random.randint(20, 40), np.random.randint(60, 100), np.random.randint(140, 195)) # BGR rust
            cv2.circle(img_with_noise, (rx, ry), rad, col, -1, lineType=cv2.LINE_AA)
            
    elif dtype == "discoloration":
        # Heat tint patina (blue/straw/purple temper ring)
        overlay = img_with_noise.copy()
        cv2.ellipse(overlay, (cx, cy), (w // 2, h // 2), params["angle"], 0, 360, (180, 120, 60), -1, lineType=cv2.LINE_AA) # bluish/purple
        cv2.ellipse(overlay, (cx, cy), (w // 2 - 15, h // 2 - 15), params["angle"], 0, 360, (60, 140, 200), -1, lineType=cv2.LINE_AA) # golden straw
        cv2.addWeighted(overlay, 0.45, img_with_noise, 0.55, 0, img_with_noise)
        
    # Convert BGR to RGB for standard image pipelines
    img_rgb = cv2.cvtColor(img_with_noise, cv2.COLOR_BGR2RGB)
    return img_rgb, specimen

def ndarray_to_base64(img_rgb: np.ndarray, format: str = "PNG") -> str:
    """Converts a numpy RGB or RGBA array to base64 data URI."""
    pil_img = Image.fromarray(img_rgb)
    buffer = io.BytesIO()
    pil_img.save(buffer, format=format)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{encoded}"

def generate_mock_heatmap(img_shape: Tuple[int, int], params: Dict[str, Any]) -> Tuple[np.ndarray, str]:
    """
    Generates a realistic 2D Grad-CAM similarity heatmap over the defect location,
    rendered with the Turbo / Jet colormap.
    """
    h, w = img_shape[:2]
    cx, cy = params["center"]
    sw, sh = params["size"]
    
    if params["type"] == "normal":
        # Diffuse uniform low-activation background
        heatmap_norm = np.zeros((h, w), dtype=np.float32)
        cv2.circle(heatmap_norm, (w // 2, h // 2), int(w * 0.35), 0.12, -1)
        heatmap_norm = cv2.GaussianBlur(heatmap_norm, (101, 101), 0)
    else:
        # Bivariate Gaussian centered on defect
        Y, X = np.ogrid[:h, :w]
        sigma_x = max(25, sw * 0.8)
        sigma_y = max(25, sh * 0.8)
        
        theta = math.radians(params["angle"])
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        
        xr = (X - cx) * cos_t + (Y - cy) * sin_t
        yr = -(X - cx) * sin_t + (Y - cy) * cos_t
        
        heatmap_raw = np.exp(-((xr**2) / (2 * sigma_x**2) + (yr**2) / (2 * sigma_y**2)))
        heatmap_norm = (heatmap_raw - heatmap_raw.min()) / (heatmap_raw.max() - heatmap_raw.min() + 1e-6)
        
    # Map to Turbo/Jet colormap
    heatmap_u8 = (heatmap_norm * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    heatmap_colored_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Create RGBA version where intensity dictates alpha
    alpha = (heatmap_norm * 200).astype(np.uint8)
    heatmap_rgba = np.dstack([heatmap_colored_rgb, alpha])
    
    return heatmap_norm, ndarray_to_base64(heatmap_rgba, "PNG")

def generate_mock_sam_mask(img_shape: Tuple[int, int], params: Dict[str, Any]) -> Tuple[np.ndarray, List[List[int]], str, int]:
    """
    Generates a crisp SAM 2 segmentation polygon mask and transparent overlay.
    """
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = params["center"]
    sw, sh = params["size"]
    
    if params["type"] == "normal":
        return mask, [], ndarray_to_base64(np.zeros((h, w, 4), dtype=np.uint8), "PNG"), 0
        
    num_pts = 16
    contour = []
    for i in range(num_pts):
        angle = (2 * math.pi / num_pts) * i
        # Organic jitter for defect contour
        r_jitter = 1.0 + np.random.uniform(-0.25, 0.25)
        rx = (sw // 2) * math.cos(angle) * r_jitter
        ry = (sh // 2) * math.sin(angle) * r_jitter
        
        # Rotate by params angle
        theta = math.radians(params["angle"])
        rx_rot = rx * math.cos(theta) - ry * math.sin(theta)
        ry_rot = rx * math.sin(theta) + ry * math.cos(theta)
        
        px = int(np.clip(cx + rx_rot, 0, w - 1))
        py = int(np.clip(cy + ry_rot, 0, h - 1))
        contour.append([px, py])
        
    contour_np = np.array(contour, dtype=np.int32)
    cv2.fillPoly(mask, [contour_np], 255)
    
    area_px = int(np.sum(mask > 0))
    
    # Create high-contrast cyan overlay (RGB: 6, 182, 212)
    mask_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    mask_rgba[mask > 0] = [6, 182, 212, 175] # semi-transparent cyan fill
    
    # Bright border
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask_rgba, contours, -1, (255, 255, 255, 255), 2, lineType=cv2.LINE_AA)
    
    return mask, contour, ndarray_to_base64(mask_rgba, "PNG"), area_px

def generate_mock_depth_map(img_shape: Tuple[int, int], params: Dict[str, Any]) -> Tuple[np.ndarray, str]:
    """
    Generates a Depth Anything V2 monocular depth map showing relative topological elevation
    (pseudo-color Inferno / Viridis).
    """
    h, w = img_shape[:2]
    cx, cy = params["center"]
    sw, sh = params["size"]
    
    # 1. Base flange depth geometry (raised center, bevel edges)
    Y, X = np.ogrid[:h, :w]
    center_x, center_y = w // 2, h // 2
    dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    
    # Normalized radial elevation (flange surface is relatively flat with chamfer drops)
    depth_base = np.clip(1.0 - (dist_from_center / (w * 0.45)), 0.0, 1.0)
    
    # 2. Inject defect topological elevation/depression
    if params["type"] in ["crack", "dent", "scratch", "corrosion"]:
        # Negative depression pit
        r_defect = np.sqrt(((X - cx) / max(1, sw))**2 + ((Y - cy) / max(1, sh))**2)
        depth_depression = np.exp(-3.0 * (r_defect**2)) * abs(params["delta_z_mm"] / 2.5)
        depth_base = np.clip(depth_base - depth_depression, 0.0, 1.0)
    elif params["type"] == "discoloration":
        # Discoloration has ZERO depth displacement (flat)
        pass
        
    depth_u8 = (depth_base * 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)
    depth_colored_rgb = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
    
    return depth_base, ndarray_to_base64(depth_colored_rgb, "PNG")
